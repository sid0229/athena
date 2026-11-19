"""Storage for reviewer corrections to extracted medication lists.

Stores each correction in local SQLite (data/corrections.db):
- note hash
- original extraction (JSON)
- corrected list (JSON)
- reviewer
- timestamp (time)
- case_id (optional reference)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from athena.config import data_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    note_hash TEXT NOT NULL,
    original_extraction TEXT NOT NULL,
    corrected_list TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    case_id TEXT,
    note_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_corrections_note ON corrections(note_hash);
"""


@dataclass(frozen=True)
class CorrectionEntry:
    id: int
    ts: str
    note_hash: str
    original_extraction: list[dict[str, Any]]
    corrected_list: list[dict[str, Any]]
    reviewer: str
    case_id: str | None = None
    note_text: str | None = None


class CorrectionsDB:
    def __init__(self, path: Path | str | None = None):
        if path:
            self.path = Path(path)
        else:
            try:
                self.path = data_path("corrections_db")
            except Exception:
                self.path = Path("data/corrections.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._con()) as con, con:
            con.executescript(SCHEMA)
            cols = {row[1] for row in con.execute("PRAGMA table_info(corrections)").fetchall()}
            if "note_text" not in cols:
                con.execute("ALTER TABLE corrections ADD COLUMN note_text TEXT")

    def _con(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, *, note_hash: str, original_extraction: list[dict] | str,
               corrected_list: list[dict] | str, reviewer: str,
               case_id: str | None = None, note_text: str | None = None) -> CorrectionEntry:
        if not reviewer.strip():
            raise ValueError("reviewer name is required")
        if not note_hash.strip():
            raise ValueError("note_hash is required")

        orig_json = original_extraction if isinstance(original_extraction, str) else json.dumps(original_extraction, ensure_ascii=False)
        corr_json = corrected_list if isinstance(corrected_list, str) else json.dumps(corrected_list, ensure_ascii=False)

        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._con()) as con, con:
            cur = con.execute(
                "INSERT INTO corrections (ts, note_hash, original_extraction, corrected_list, reviewer, case_id, note_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, note_hash, orig_json, corr_json, reviewer.strip(), case_id, note_text)
            )
            cid = cur.lastrowid

        orig_parsed = json.loads(orig_json) if isinstance(orig_json, str) else original_extraction
        corr_parsed = json.loads(corr_json) if isinstance(corr_json, str) else corrected_list
        return CorrectionEntry(
            id=cid,
            ts=ts,
            note_hash=note_hash,
            original_extraction=orig_parsed,
            corrected_list=corr_parsed,
            reviewer=reviewer.strip(),
            case_id=case_id,
            note_text=note_text,
        )

    def entries(self, note_hash: str | None = None) -> list[CorrectionEntry]:
        q = "SELECT id, ts, note_hash, original_extraction, corrected_list, reviewer, case_id, note_text FROM corrections"
        params = ()
        if note_hash:
            q += " WHERE note_hash = ?"
            params = (note_hash,)
        q += " ORDER BY id"
        with closing(self._con()) as con:
            rows = con.execute(q, params).fetchall()
        return [
            CorrectionEntry(
                id=r[0],
                ts=r[1],
                note_hash=r[2],
                original_extraction=json.loads(r[3]),
                corrected_list=json.loads(r[4]),
                reviewer=r[5],
                case_id=r[6],
                note_text=r[7] if len(r) > 7 else None,
            )
            for r in rows
        ]

    def latest(self, note_hash: str) -> CorrectionEntry | None:
        ents = self.entries(note_hash)
        return ents[-1] if ents else None

    def count(self) -> int:
        with closing(self._con()) as con:
            row = con.execute("SELECT COUNT(*) FROM corrections").fetchone()
            return row[0] if row else 0
