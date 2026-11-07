"""Append-only, tamper-evident audit log for pharmacist decisions (README §2.5).

Every confirm / override is a new row; nothing is ever updated or deleted
(SQLite triggers reject UPDATE and DELETE). Rows are hash-chained:

    row_hash = sha256(prev_hash || canonical JSON of the row's fields)

so any later edit to the file breaks `verify()`. Each row also stores the
SHA-256 of the system report the reviewer was looking at.

The chain alone cannot see rows removed from the *end* (or a fully rewritten file), so the
row count and head hash are also written to a sidecar file (`<db>.head`) after every append and
checked by `verify()`; timestamps must never go backwards. (Code review, Nov 2026.)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from athena.config import data_path

ACTIONS = ("confirm", "override")
OVERRIDE_REASONS = {
    "not_clinically_relevant": "Not clinically relevant for this patient",
    "already_managed": "Already managed / monitored",
    "extraction_error": "Extraction error (wrong drug, dose or status)",
    "intended_change": "Change is intended",
    "other": "Other (see comment)",
}
GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    case_id TEXT NOT NULL,
    report_hash TEXT NOT NULL,
    finding_id TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    title TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('confirm', 'override')),
    reason_code TEXT,
    comment TEXT,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL UNIQUE
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE INDEX IF NOT EXISTS audit_case ON audit(case_id, finding_id);
"""

FIELDS = ("ts", "reviewer", "case_id", "report_hash", "finding_id", "finding_type", "tier",
          "title", "action", "reason_code", "comment")


@dataclass(frozen=True)
class AuditEntry:
    id: int
    ts: str
    reviewer: str
    case_id: str
    report_hash: str
    finding_id: str
    finding_type: str
    tier: str
    title: str
    action: str
    reason_code: str | None
    comment: str | None
    prev_hash: str
    row_hash: str


def _row_hash(prev_hash: str, fields: dict) -> str:
    payload = json.dumps({k: fields.get(k) for k in FIELDS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((prev_hash + payload).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else data_path("audit_db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor = self.path.with_name(self.path.name + ".head")
        with closing(self._con()) as con, con:
            con.executescript(SCHEMA)
        if not self.anchor.exists():  # first use, or a log created before anchoring existed
            self._write_anchor()

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _state(self) -> tuple[int, str]:
        with closing(self._con()) as con:
            n, = con.execute("SELECT COUNT(*) FROM audit").fetchone()
            head = con.execute("SELECT row_hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return n, head[0] if head else GENESIS

    def _write_anchor(self) -> None:
        n, head = self._state()
        tmp = self.anchor.with_name(self.anchor.name + ".tmp")
        tmp.write_text(json.dumps({"rows": n, "head": head}))
        tmp.replace(self.anchor)

    def record(self, *, reviewer: str, case_id: str, report_hash: str, finding_id: str,
               finding_type: str, tier: str, title: str, action: str,
               reason_code: str | None = None, comment: str | None = None) -> AuditEntry:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        if action == "override":
            if reason_code not in OVERRIDE_REASONS:
                raise ValueError("an override needs a reason code")
            if reason_code == "other" and not (comment or "").strip():
                raise ValueError("reason 'other' needs a comment")
        fields = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reviewer": reviewer.strip(), "case_id": case_id, "report_hash": report_hash,
            "finding_id": finding_id, "finding_type": finding_type, "tier": tier, "title": title,
            "action": action, "reason_code": reason_code if action == "override" else None,
            "comment": (comment or "").strip() or None,
        }
        ok, bad = self.verify()
        if not ok:
            raise ValueError(f"audit log failed verification at row {bad}; not appending to a broken chain")
        with closing(self._con()) as con, con:
            con.execute("BEGIN IMMEDIATE")
            prev = con.execute("SELECT row_hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev_hash = prev[0] if prev else GENESIS
            row_hash = _row_hash(prev_hash, fields)
            cur = con.execute(
                f"INSERT INTO audit ({', '.join(FIELDS)}, prev_hash, row_hash) "
                f"VALUES ({', '.join('?' * (len(FIELDS) + 2))})",
                [fields[k] for k in FIELDS] + [prev_hash, row_hash])
            rid = cur.lastrowid
        self._write_anchor()
        return AuditEntry(rid, **fields, prev_hash=prev_hash, row_hash=row_hash)

    def entries(self, case_id: str | None = None) -> list[AuditEntry]:
        q = "SELECT * FROM audit" + (" WHERE case_id = ?" if case_id else "") + " ORDER BY id"
        with closing(self._con()) as con:
            return [AuditEntry(*r) for r in con.execute(q, (case_id,) if case_id else ())]

    def latest_decisions(self, case_id: str, report_hash: str) -> dict[str, AuditEntry]:
        """Most recent decision per finding for this exact report."""
        out = {}
        for e in self.entries(case_id):
            if e.report_hash == report_hash:
                out[e.finding_id] = e
        return out

    def verify(self) -> tuple[bool, int | None]:
        """(True, None) if intact, else (False, first bad row id).

        Checks the hash chain, that timestamps never go backwards, and that the row count and
        head hash match the sidecar anchor (catches rows deleted from the end)."""
        prev, last_ts, entries = GENESIS, "", self.entries()
        for e in entries:
            fields = {k: getattr(e, k) for k in FIELDS}
            if e.prev_hash != prev or e.row_hash != _row_hash(prev, fields) or e.ts < last_ts:
                return False, e.id
            prev, last_ts = e.row_hash, e.ts
        if self.anchor.exists():
            a = json.loads(self.anchor.read_text())
            if a.get("rows") != len(entries) or a.get("head") != prev:
                return False, (entries[-1].id + 1) if entries else 1
        return True, None
