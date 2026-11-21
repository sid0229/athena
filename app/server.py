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

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from athena.config import load_config  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze, analyze_medications  # noqa: E402
from athena.ingest import MAX_BYTES, IngestError, extract_document, extract_text  # noqa: E402
from athena.review.audit import OVERRIDE_REASONS, AuditLog  # noqa: E402
from athena.review.corrections import CorrectionsDB  # noqa: E402
from athena.verification.checker import InteractionChecker  # noqa: E402

VERSION = "0.4"
DEMO_DIR = ROOT / "data" / "demo"
DIST = ROOT / "web" / "dist"
MODES = {"hybrid": "Hybrid — local LLM + rules", "rules": "Rules only", "llm": "LLM only"}
PRIOR_LIST_HEADER = "Medications on Admission:"

app = FastAPI(title="Athena", docs_url=None, redoc_url=None)
# Only answer requests addressed to this machine (blocks DNS-rebinding pages in the browser).
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


# ---------------------------------------------------------------- shared resources (unchanged backend)

@lru_cache(maxsize=1)
def checker() -> InteractionChecker:
    return InteractionChecker()


@lru_cache(maxsize=3)
def extractor(mode: str) -> MedicationExtractor:
    return MedicationExtractor(mode, normalizer=checker().normalizer)


@lru_cache(maxsize=1)
def predictor():
    """XGBoost research model, loaded on first use (only for pairs no database covers)."""
    from athena.verification.ddi_predictor import InteractionPredictor
    return InteractionPredictor()


def ml_signals(r: dict) -> list[dict]:
    """Unverified ML probabilities for NOT_COVERED pairs. Display only: not in the report, the risk
    tiers or the audit log."""
    f = next((x for x in r["findings"] if x["type"] == "NOT_COVERED"), None)
    out = []
    for p in (f or {}).get("evidence", {}).get("pairs", []):
        try:
            pred = predictor().predict(p["a"], p["b"])
        except Exception:  # model files missing: no signal, never an error for the reviewer
            return []
        if pred:
            out.append({**p, **pred})
    return sorted(out, key=lambda x: -x["probability"])


@lru_cache(maxsize=1)
def audit() -> AuditLog:
    return AuditLog(os.environ.get("ATHENA_AUDIT_DB") or None)


@lru_cache(maxsize=1)
def corrections() -> CorrectionsDB:
    return CorrectionsDB(os.environ.get("ATHENA_CORRECTIONS_DB") or None)


_cache: OrderedDict[tuple[str, str, str, str | None], dict] = OrderedDict()   # (note sha, mode, source, ocr_engine) -> report
_reports: OrderedDict[str, dict] = OrderedDict()              # report_hash -> {report, text, case_id}
_lock = Lock()        # one analysis at a time: the 8 GB laptop runs a single local model
_cache_lock = Lock()  # cache dicts are touched from FastAPI's worker threads


def run_analysis(text: str, mode: str, input_source: str = "digital", ocr_engine: str | None = None) -> dict:
    key = (hashlib.sha256(text.encode()).hexdigest(), mode, input_source, ocr_engine)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            r = _cache[key]
            _reports.setdefault(r["meta"]["report_hash"], {"report": r, "text": text, "case_id": case_id_for(text)})
            return r
    with _lock:
        r = analyze(text, extractor(mode), checker(), input_source=input_source, ocr_engine=ocr_engine).to_dict()
    with _cache_lock:
        # A run that fell back to rules (LLM offline) is not cached under the requested mode.
        if r["meta"]["extraction_mode"] == mode:
            _cache[key] = r
            if len(_cache) > 32:
                _cache.popitem(last=False)
        _reports[r["meta"]["report_hash"]] = {"report": r, "text": text, "case_id": case_id_for(text)}
        if len(_reports) > 256:
            _reports.popitem(last=False)
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
    input_source: str = "digital"     # "digital" | "ocr"
    ocr_engine: str | None = None     # "vision" | "qwen_vl" | "tesseract"


class DecisionIn(BaseModel):
    report_hash: str          # the exact report the reviewer was looking at
    finding_id: str
    reviewer: str
    action: str
    reason_code: str | None = None
    comment: str | None = None


class CorrectIn(BaseModel):
    report_hash: str
    reviewer: str
    medications: list[dict]
    comment: str | None = None
    text: str | None = None
    input_source: str = "digital"
    ocr_engine: str | None = None


@app.get("/api/meta")
def meta():
    kb = dict(checker().kb.con.execute("SELECT key, value FROM meta"))
    demos = [{"id": p.stem, "name": _pretty_title(p.stem.split("_", 1)[-1]), "text": p.read_text()}
             for p in sorted(DEMO_DIR.glob("*.txt"))]
    ocr_cfg = load_config().get("ocr", {})
    return {"version": VERSION, "model": extractor("hybrid").llm.model, "llm_online": llm_online(),
            "kb": {"concepts": int(kb["concepts"]), "pairs": int(kb["pairs"]), "built": kb.get("built"),
                   "sources": kb.get("sources")},
            "modes": MODES, "override_reasons": OVERRIDE_REASONS, "demos": demos,
            "ocr": {"engine": ocr_cfg.get("engine", "vision"), "low_confidence_threshold": ocr_cfg.get("low_confidence_threshold", 0.70)}}


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
    r = run_analysis(text, body.mode, input_source=body.input_source, ocr_engine=body.ocr_engine)
    cid = case_id_for(text)
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    return {"report": r, "text": text, "case_id": cid, "display_names": display_names(r), "ml_signals": ml_signals(r),
            "decisions": {k: _entry(v) for k, v in decisions.items()}}


@app.post("/api/extract")
async def api_extract(request: Request, filename: str, ocr_engine: str | None = None):
    """Raw file bytes in the body (no multipart dependency) -> note text, source, OCR details."""
    data = b""
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "File is larger than 20 MB.")
    try:
        doc = await run_in_threadpool(extract_document, filename, data, ocr_engine)
        return {
            "name": filename,
            "text": doc.text,
            "source": doc.source,
            "ocr_engine": doc.ocr_engine,
            "low_confidence_words": doc.low_confidence_words,
            "page_count": doc.page_count,
            "seconds": doc.seconds,
            "preview_images": doc.preview_images,
        }
    except IngestError as e:
        raise HTTPException(400, str(e))


@app.post("/api/decision")
def api_decision(body: DecisionIn):
    with _cache_lock:
        saved = _reports.get(body.report_hash)
    if saved is None:  # never recompute: a new run could differ from what the reviewer saw
        raise HTTPException(409, "This report is no longer loaded on the server — analyze the note again.")
    r, cid = saved["report"], saved["case_id"]
    f = next((x for x in r["findings"] if x["id"] == body.finding_id), None)
    if f is None:
        raise HTTPException(404, "finding not in this report")
    try:
        audit().record(reviewer=body.reviewer, case_id=cid, report_hash=r["meta"]["report_hash"],
                       finding_id=f["id"], finding_type=f["type"], tier=f["tier"], title=f["title"],
                       action=body.action, reason_code=body.reason_code, comment=body.comment)
    except ValueError as e:
        raise HTTPException(400, str(e))
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    return {"decisions": {k: _entry(v) for k, v in decisions.items()}}


@app.post("/api/correct")
def api_correct(body: CorrectIn):
    if not body.reviewer or not body.reviewer.strip():
        raise HTTPException(400, "reviewer name is required")
    with _cache_lock:
        saved = _reports.get(body.report_hash)
    if saved is not None:
        text = saved["text"]
        cid = saved["case_id"]
        orig_meds = saved["report"]["medications"]
    elif body.text and body.text.strip():
        text = body.text
        cid = case_id_for(text)
        orig_meds = []
    else:
        raise HTTPException(409, "This report is no longer loaded on the server — analyze the note again.")

    with _lock:
        r = analyze_medications(body.medications, text=text, checker=checker(),
                                input_source=body.input_source, ocr_engine=body.ocr_engine).to_dict()

    note_hash = hashlib.sha256(text.encode()).hexdigest()
    corrections().record(
        note_hash=note_hash,
        original_extraction=orig_meds,
        corrected_list=r["medications"],
        reviewer=body.reviewer.strip(),
        case_id=cid,
        note_text=text,
    )

    try:
        audit().record(
            reviewer=body.reviewer.strip(),
            case_id=cid,
            report_hash=r["meta"]["report_hash"],
            finding_id="CORRECTION",
            finding_type="MEDICATION_CORRECTION",
            tier="info",
            title=f"Medication list corrected ({len(r['medications'])} medications)",
            action="correct",
            comment=body.comment,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    with _cache_lock:
        _reports[r["meta"]["report_hash"]] = {"report": r, "text": text, "case_id": cid}
        if len(_reports) > 256:
            _reports.popitem(last=False)

    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    return {
        "report": r,
        "text": text,
        "case_id": cid,
        "display_names": display_names(r),
        "ml_signals": ml_signals(r),
        "decisions": {k: _entry(v) for k, v in decisions.items()},
    }


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
        w.writerow([_csv_safe(x) for x in (e.ts, e.reviewer, e.case_id, e.tier, e.title, e.action,
                    OVERRIDE_REASONS.get(e.reason_code, "") if e.reason_code else "", e.comment or "",
                    e.report_hash[:12], e.row_hash)])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=athena_audit.csv"})


def _csv_safe(v: str) -> str:
    """Spreadsheet apps execute cells starting with = + - @ (formula injection): prefix a quote."""
    return "'" + v if v and v[0] in "=+-@\t\r" else v


@app.get("/api/status")
def api_status():
    return {"llm_online": llm_online()}


# ---------------------------------------------------------------- built web app

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    _DIST = DIST.resolve()

    @app.get("/{path:path}")
    def spa(path: str):
        f = (DIST / path).resolve()
        if not f.is_relative_to(_DIST):  # "/../../configs/..." must never leave web/dist
            raise HTTPException(404)
        return FileResponse(f if path and f.is_file() else DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ATHENA_PORT", 8000)))
