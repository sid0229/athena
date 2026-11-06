"""Athena — local API for the web dashboard (web/).

    python app/server.py            # http://localhost:8000 (serves web/dist if built)

Presentation layer only: it calls the existing pipeline (`analyze`), checker and audit
log unchanged and returns their results as JSON. Binds to localhost; no patient text
leaves the device.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze  # noqa: E402
from athena.review.audit import OVERRIDE_REASONS, AuditLog  # noqa: E402
from athena.verification.checker import InteractionChecker  # noqa: E402

VERSION = "0.4"
DEMO_DIR = ROOT / "data" / "demo"
DIST = ROOT / "web" / "dist"
MODES = {"hybrid": "Hybrid — local LLM + rules", "rules": "Rules only", "llm": "LLM only"}
PRIOR_LIST_HEADER = "Medications on Admission:"

app = FastAPI(title="Athena", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------- shared resources (unchanged backend)

@lru_cache(maxsize=1)
def checker() -> InteractionChecker:
    return InteractionChecker()


@lru_cache(maxsize=3)
def extractor(mode: str) -> MedicationExtractor:
    return MedicationExtractor(mode, normalizer=checker().normalizer)


@lru_cache(maxsize=1)
def audit() -> AuditLog:
    return AuditLog(os.environ.get("ATHENA_AUDIT_DB") or None)


_cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
_lock = Lock()  # one analysis at a time: the 8 GB laptop runs a single local model


def run_analysis(text: str, mode: str) -> dict:
    key = (hashlib.sha256(text.encode()).hexdigest(), mode)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    with _lock:
        r = analyze(text, extractor(mode), checker()).to_dict()
    _cache[key] = r
    if len(_cache) > 32:
        _cache.popitem(last=False)
    return r


def case_id_for(text: str) -> str:
    return "case-" + hashlib.sha256(text.encode()).hexdigest()[:10]


def _pretty_title(stem: str) -> str:
    words = stem.replace("_", " ").split()
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in words)


@lru_cache(maxsize=1)
def common_names() -> dict[str, str]:
    """Display only: DrugBank/international name -> everyday US name from RxNorm
    (acetylsalicylic acid -> aspirin). Matching and interaction logic never use this."""
    import pandas as pd

    from athena.normalize.lexicon import load_lexicon, norm
    rrf = ROOT / "data" / "raw" / "rxnorm" / "rrf" / "RXNCONSO.RRF"
    if not rrf.exists():
        return {}
    con = pd.read_csv(rrf, sep="|", header=None, dtype=str, usecols=[11, 12, 14], names=["sab", "tty", "str"])
    ins = set(con[(con["sab"] == "RXNORM") & (con["tty"] == "IN")]["str"].map(norm))
    cands: dict[str, list[str]] = {}
    for surface, targets in load_lexicon()["index"].items():
        if len(targets) != 1 or surface not in ins:
            continue
        canon = targets[0]
        if surface == canon or canon in ins:
            continue
        if canon.startswith(surface + " ") or surface.startswith(canon + " "):
            continue
        if "," in surface or surface.startswith(("p-", "alum")):
            continue
        cands.setdefault(canon, []).append(surface)
    return {c: min(v, key=len) for c, v in cands.items()}


def display_names(r: dict) -> dict[str, str]:
    """Common names for every drug name that appears in the report."""
    names = set()
    for m in r["medications"]:
        names.update(p.strip().lower() for p in m["name"].split(" + "))
    for f in r["findings"]:
        names.update(p.strip().lower() for p in f["title"].split(": ", 1)[-1].split(" + "))
    cn = common_names()
    return {n: cn[n] for n in names if n in cn}


def _entry(e) -> dict:
    return {k: getattr(e, k) for k in ("id", "ts", "reviewer", "case_id", "report_hash", "finding_id",
                                       "finding_type", "tier", "title", "action", "reason_code", "comment",
                                       "prev_hash", "row_hash")}


def llm_online() -> bool:
    try:
        return extractor("hybrid").llm.available()
    except Exception:
        return False


# ---------------------------------------------------------------- API

class AnalyzeIn(BaseModel):
    text: str
    mode: str = "hybrid"
    prior_list: str | None = None


class DecisionIn(BaseModel):
    text: str
    mode: str = "hybrid"
    finding_id: str
    reviewer: str
    action: str
    reason_code: str | None = None
    comment: str | None = None


@app.get("/api/meta")
def meta():
    kb = dict(checker().kb.con.execute("SELECT key, value FROM meta"))
    demos = [{"id": p.stem, "name": _pretty_title(p.stem.split("_", 1)[-1]), "text": p.read_text()}
             for p in sorted(DEMO_DIR.glob("*.txt"))]
    return {"version": VERSION, "model": extractor("hybrid").llm.model, "llm_online": llm_online(),
            "kb": {"concepts": int(kb["concepts"]), "pairs": int(kb["pairs"]), "built": kb.get("built"),
                   "sources": kb.get("sources")},
            "modes": MODES, "override_reasons": OVERRIDE_REASONS, "demos": demos}


@app.post("/api/analyze")
def api_analyze(body: AnalyzeIn):
    if body.mode not in MODES:
        raise HTTPException(400, "unknown mode")
    text = body.text
    if body.prior_list and body.prior_list.strip():
        # Same path as a note that already contains both lists (see the Streamlit app).
        text = f"{PRIOR_LIST_HEADER}\n{body.prior_list.strip()}\n\nDischarge Summary:\n{body.text}"
    if not text.strip():
        raise HTTPException(400, "empty note")
    r = run_analysis(text, body.mode)
    cid = case_id_for(text)
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    return {"report": r, "text": text, "case_id": cid, "display_names": display_names(r),
            "decisions": {k: _entry(v) for k, v in decisions.items()}}


@app.post("/api/decision")
def api_decision(body: DecisionIn):
    r = run_analysis(body.text, body.mode)
    f = next((x for x in r["findings"] if x["id"] == body.finding_id), None)
    if f is None:
        raise HTTPException(404, "finding not in this report")
    cid = case_id_for(body.text)
    try:
        audit().record(reviewer=body.reviewer, case_id=cid, report_hash=r["meta"]["report_hash"],
                       finding_id=f["id"], finding_type=f["type"], tier=f["tier"], title=f["title"],
                       action=body.action, reason_code=body.reason_code, comment=body.comment)
    except ValueError as e:
        raise HTTPException(400, str(e))
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    return {"decisions": {k: _entry(v) for k, v in decisions.items()}}


@app.get("/api/audit")
def api_audit():
    ok, bad = audit().verify()
    entries = [_entry(e) for e in reversed(audit().entries())]
    return {"chain_ok": ok, "broken_at": bad, "entries": entries}


@app.get("/api/audit.csv")
def api_audit_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time_utc", "reviewer", "case", "tier", "finding", "action", "reason", "comment", "report", "row_hash"])
    for e in reversed(audit().entries()):
        w.writerow([e.ts, e.reviewer, e.case_id, e.tier, e.title, e.action,
                    OVERRIDE_REASONS.get(e.reason_code, "") if e.reason_code else "", e.comment or "",
                    e.report_hash[:12], e.row_hash])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=athena_audit.csv"})


@app.get("/api/status")
def api_status():
    return {"llm_online": llm_online()}


# ---------------------------------------------------------------- built web app

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ATHENA_PORT", 8000)))
