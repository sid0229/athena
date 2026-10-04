"""Athena — pharmacist review dashboard (Batch 6).

    streamlit run app/streamlit_app.py

Runs fully on this machine: local LLM via Ollama on localhost, local SQLite KB and
audit log, fonts served from app/static. No patient text leaves the device.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze  # noqa: E402
from athena.review.audit import OVERRIDE_REASONS, AuditLog  # noqa: E402
from athena.verification.checker import InteractionChecker  # noqa: E402
from ui.theme import TIERS, TYPE_LABELS, chip, css, esc, icon, meter, tier_badge  # noqa: E402

st.set_page_config(page_title="Athena · Medication Safety", page_icon=":material/medication:",
                   layout="wide", initial_sidebar_state="expanded")
st.html(css())
st.logo(str(Path(__file__).parent / "static" / "athena-logo.svg"), size="large")

DEMO_DIR = ROOT / "data" / "demo"
MODES = {"hybrid": "Hybrid — local LLM + rules", "rules": "Rules only — fast, no LLM"}


# ---------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner=False)
def checker() -> InteractionChecker:
    return InteractionChecker()


@st.cache_resource(show_spinner=False)
def extractor(mode: str) -> MedicationExtractor:
    return MedicationExtractor(mode, normalizer=checker().normalizer)


@st.cache_resource(show_spinner=False)
def audit() -> AuditLog:
    # ATHENA_AUDIT_DB lets demos / automated screenshots write to a separate file.
    return AuditLog(os.environ.get("ATHENA_AUDIT_DB") or None)


@st.cache_data(show_spinner=False, ttl=15)
def llm_online() -> bool:
    try:
        return extractor("hybrid").llm.available()
    except Exception:
        return False


@st.cache_data(show_spinner=False, max_entries=32)
def run_analysis(text: str, mode: str) -> dict:
    return analyze(text, extractor(mode), checker()).to_dict()


def kb_meta() -> dict:
    return dict(checker().kb.con.execute("SELECT key, value FROM meta"))


def case_id_for(text: str) -> str:
    return "case-" + hashlib.sha256(text.encode()).hexdigest()[:10]


def demo_notes() -> dict[str, str]:
    out = {}
    for p in sorted(DEMO_DIR.glob("*.txt")):
        title = p.stem.split("_", 1)[-1].replace("_", " ").title()
        out[f"{title}  ·  synthetic"] = p.read_text()
    return out


# ---------------------------------------------------------------- sidebar

def sidebar():
    with st.sidebar:
        st.html('<div class="side-label">Reviewer</div>')
        st.text_input("Reviewer name", key="reviewer", placeholder="e.g. Dr. A. Sharma, PharmD",
                      label_visibility="collapsed")
        st.html('<div class="side-label">Extraction</div>')
        st.selectbox("Extraction mode", list(MODES), format_func=MODES.get, key="mode",
                     label_visibility="collapsed")
        st.html('<div class="side-label">System</div>')
        online = llm_online()
        meta = kb_meta()
        model = extractor("hybrid").llm.model
        st.html(f"""
        <div class="status-row"><span class="dot {'dot-ok' if online else 'dot-off'}"></span>
          {icon("cpu", 14)} Local LLM · {esc(model)} {'online' if online else 'offline'}</div>
        <div class="status-row"><span class="dot dot-ok"></span>{icon("database", 14)}
          KB · {int(meta['concepts']):,} drugs · {int(meta['pairs']):,} pairs</div>
        <div class="status-row"><span class="dot dot-ok"></span>{icon("lock", 14)} On-device · no network calls</div>
        <div class="side-foot">Decision support only. Interaction flags come from DrugBank and DDInter
        records — never from the language model. A pharmacist confirms every finding.</div>""")


# ---------------------------------------------------------------- review page

def note_input():
    st.html("""<div class="page-head"><div>
      <div class="eyebrow">Medication reconciliation</div>
      <div class="page-title">Review a discharge summary</div>
      <div class="page-sub">Athena extracts the medication list, reconciles admission vs discharge and
      checks every active pair against verified interaction databases.</div></div></div>""")
    t1, t2, t3 = st.tabs(["Demo patients", "Paste note", "Upload .txt"])
    text = None
    with t1:
        notes = demo_notes()
        choice = st.selectbox("Synthetic demo patient", list(notes), label_visibility="collapsed")
        st.html('<div class="callout">' + icon("info", 16) +
                "<div>Demo notes are fictional, written for this prototype. Use them for screen-sharing; "
                "real clinical notes must stay on this device.</div></div>")
        if st.button("Analyze demo note", type="primary", key="go-demo"):
            text = notes[choice]
    with t2:
        pasted = st.text_area("Clinical note", height=260, placeholder="Paste a discharge summary…",
                              label_visibility="collapsed")
        if st.button("Analyze note", type="primary", key="go-paste", disabled=not pasted.strip()):
            text = pasted
    with t3:
        up = st.file_uploader("Upload a plain-text note", type=["txt"], label_visibility="collapsed")
        if st.button("Analyze upload", type="primary", key="go-up", disabled=up is None):
            text = up.read().decode("utf-8", errors="replace")
    if text:
        st.session_state.note = text
        st.rerun()


def kpis(r: dict):
    c = r["meta"]["counts"]
    tiers = c["tiers"]
    act = sum(f["tier"] in ("critical", "review") for f in r["findings"])
    tiles = [
        ("k-critical", "octagon-alert", "Critical", tiers["critical"], "act before discharge"),
        ("k-review", "triangle-alert", "Review", tiers["review"], "pharmacist judgement"),
        ("k-info", "info", "Info", tiers["info"], "shown, low priority"),
        ("k-accent", "pill", "Active meds", c["active"], f"{c['medications']} drugs in note"),
        ("", "arrow-right-left", "Pairs checked", c["pairs_checked"],
         f"{c['interactions']} with a record · {c['not_covered']} not covered"),
    ]
    html_tiles = "".join(
        f'<div class="kpi {k}"><div class="kpi-label">{icon(ic, 14)}{esc(lbl)}</div>'
        f'<div class="kpi-value">{v:,}</div><div class="kpi-foot">{esc(foot)}</div></div>'
        for k, ic, lbl, v, foot in tiles)
    st.html(f'<div class="kpis" role="status" aria-label="{act} findings need action">{html_tiles}</div>')


def finding_card(f: dict, idx: int, r: dict, decisions: dict):
    tier = f["tier"]
    tlabel, ticon = TYPE_LABELS.get(f["type"], (f["type"], "info"))
    with st.container(key=f"card-{tier}-{idx}"):
        why = "".join(f"<li>{esc(line)}</li>" for line in f["explanation"])
        verify = (f'<div class="verify">{icon("triangle-alert", 14)} Possible Major interaction — verify the '
                  f'extracted drugs</div>') if f.get("needs_verification") else ""
        risk = meter(f["risk"], "risk") if f["type"] == "INTERACTION" else ""
        srcs = ""
        ev = f.get("evidence") or {}
        if ev.get("sources"):
            srcs = "".join(chip(s.upper() if s == "ddinter" else "DrugBank", "accent", "database")
                           for s in ev["sources"])
            if ev.get("severity_basis") == "derived":
                srcs += chip("severity derived", "warn")
        st.html(f"""
        <div class="f-head">{tier_badge(tier)}<span class="f-type">{icon(ticon, 13)}{esc(tlabel)}</span>
          <span class="f-risk">{('risk ' + risk) if risk else ''}</span></div>
        <div class="f-title">{esc(f['title'])}</div>{verify}
        <div>{srcs}</div>
        <ul class="f-why">{why}</ul>""")

        if f["type"] in ("NOT_CHECKED", "NOT_COVERED"):
            return
        d = decisions.get(f["id"])
        if d:
            if d.action == "confirm":
                st.html(f'<div class="decision d-confirm">{icon("circle-check", 15)} Confirmed by '
                        f'<b>{esc(d.reviewer)}</b> <small>{esc(_ts(d.ts))}</small></div>')
            else:
                st.html(f'<div class="decision d-override">{icon("undo", 15)} Overridden by '
                        f'<b>{esc(d.reviewer)}</b> — {esc(OVERRIDE_REASONS[d.reason_code])}'
                        f'{(": " + esc(d.comment)) if d.comment else ""} '
                        f'<small>{esc(_ts(d.ts))}</small></div>')
        reviewer = st.session_state.get("reviewer", "").strip()
        c1, c2, _ = st.columns([1, 1, 1.6])
        with c1:
            if st.button("Confirm", key=f"ok-{idx}", icon=":material/check:", disabled=not reviewer,
                         help=None if reviewer else "Enter your name in the sidebar first",
                         use_container_width=True):
                _record(r, f, "confirm")
        with c2:
            with st.popover("Override", icon=":material/undo:", disabled=not reviewer,
                            use_container_width=True):
                reason = st.selectbox("Reason", list(OVERRIDE_REASONS), format_func=OVERRIDE_REASONS.get,
                                      key=f"rs-{idx}")
                note = st.text_input("Comment" + (" (required)" if reason == "other" else " (optional)"),
                                     key=f"cm-{idx}")
                if st.button("Record override", key=f"ov-{idx}", type="primary",
                             disabled=reason == "other" and not note.strip()):
                    _record(r, f, "override", reason, note)


def _ts(iso: str) -> str:
    return iso.replace("T", " ").replace("+00:00", "") + " UTC"


def _record(r: dict, f: dict, action: str, reason: str | None = None, comment: str | None = None):
    audit().record(reviewer=st.session_state.reviewer, case_id=case_id_for(st.session_state.note),
                   report_hash=r["meta"]["report_hash"], finding_id=f["id"], finding_type=f["type"],
                   tier=f["tier"], title=f["title"], action=action, reason_code=reason, comment=comment)
    st.toast(f"{'Confirmed' if action == 'confirm' else 'Override recorded'} · audit log updated",
             icon=":material/verified:")
    st.rerun()


def medication_table(r: dict):
    rows = []
    for m in r["medications"]:
        a = m["attributes"]
        dose = " · ".join(x for x in (a.get("Strength"), a.get("Route"), a.get("Frequency")) if x)
        lists = (chip("Admission", "neutral") if m["in_admission"] else "") + \
                (chip("Discharge", "accent") if m["in_discharge"] else "")
        norm = m["norm_status"].replace("_", " ")
        status = f'<span class="st-{m["status"]}"><b>{esc(m["status"].title())}</b></span>'
        if m["text_status"]:
            status += f'<div class="med-raw">{esc(m["text_status"])}</div>'
        rows.append(f"""<tr>
          <td><div class="med-name">{esc(m['name'])}</div>
              <div class="med-raw">“{esc(m['raw'])}” · {esc(norm)}</div>
              {('<div class="med-dose">' + esc(dose) + '</div>') if dose else ''}
              <div class="med-lists">{lists}</div></td>
          <td>{status}</td>
          <td>{meter(m['confidence'], 'extraction confidence')}</td></tr>""")
    st.html(f"""<div class="tabwrap"><table class="medtab" aria-label="Extracted medications">
      <thead><tr><th>Medication</th><th>Status</th><th>Confidence</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>""")


def note_view(r: dict, text: str):
    hot = set()
    for f in r["findings"]:
        if f["tier"] == "critical":
            hot.update(f["meds"])
    spans = sorted((mm["start"], mm["end"], m["id"], m["name"]) for m in r["medications"] for mm in m["mentions"])
    out, pos = [], 0
    for s, e, mid, name in spans:
        if s < pos:
            continue
        out.append(esc(text[pos:s]))
        out.append(f'<mark class="d{" hot" if mid in hot else ""}" title="{esc(name)}">{esc(text[s:e])}</mark>')
        pos = e
    out.append(esc(text[pos:]))
    st.html(f'<div class="legend"><span><mark class="d" style="padding:0 4px">drug</mark> extracted</span>'
            f'<span><mark class="d hot" style="padding:0 4px">drug</mark> in a critical finding</span></div>'
            f'<div class="note" tabindex="0" aria-label="Clinical note with extracted drugs highlighted">{"".join(out)}</div>')


def run_details(r: dict):
    m = r["meta"]
    secs = ", ".join(f"{s['kind']} ({s['title']})" for s in m["sections"]) or "none found"
    for w in m["warnings"]:
        st.html(f'<div class="callout warn">{icon("triangle-alert", 16)}<div>{esc(w)}</div></div>')
    st.html(f"""<table class="medtab"><tbody>
      <tr><td><b>Extraction mode</b></td><td>{esc(m['extraction_mode'])}</td></tr>
      <tr><td><b>Local model</b></td><td>{esc(m.get('llm_model') or '—')}</td></tr>
      <tr><td><b>Processing time</b></td><td>{m['extraction_seconds']} s</td></tr>
      <tr><td><b>Medication sections</b></td><td>{esc(secs)}</td></tr>
      <tr><td><b>No known interaction</b></td><td>{m['counts']['no_known_interaction']} pairs</td></tr>
      <tr><td><b>Report hash</b></td><td style="font-family:var(--font-mono);font-size:.76rem;word-break:break-all">{esc(m['report_hash'])}</td></tr>
    </tbody></table>""")


def _apply_query_params():
    """Shortcuts for demos: ?demo=1..N loads a synthetic note, ?mode=rules|hybrid, ?reviewer=Name."""
    qp = st.query_params
    if "mode" in qp and qp["mode"] in MODES:
        st.session_state.mode = qp["mode"]
    if "reviewer" in qp and not st.session_state.get("reviewer"):
        st.session_state.reviewer = qp["reviewer"]
    if "demo" in qp and not st.session_state.get("_demo_loaded"):
        notes = list(demo_notes().values())
        i = int(qp["demo"]) - 1 if qp["demo"].isdigit() else 0
        if 0 <= i < len(notes):
            st.session_state.note = notes[i]
            st.session_state._demo_loaded = True


def review_page():
    _apply_query_params()
    sidebar()
    text = st.session_state.get("note")
    if not text:
        note_input()
        return
    mode = st.session_state.get("mode", "hybrid")
    with st.spinner("Reading the note on this device and checking every pair…"):
        r = run_analysis(text, mode)
    cid = case_id_for(text)
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    actionable = [f for f in r["findings"] if f["tier"] in ("critical", "review")]
    done = sum(f["id"] in decisions for f in actionable)

    head_l, head_r = st.columns([3, 1.3], vertical_alignment="bottom")
    with head_l:
        st.html(f"""<div class="eyebrow">Review · {esc(cid)}</div>
          <div class="page-title">Medication safety report</div>
          <div class="page-sub">{done} of {len(actionable)} actionable findings reviewed</div>""")
    with head_r:
        st.html(f'<div style="text-align:right"><span class="hash">{icon("link", 13)}'
                f'report {esc(r["meta"]["report_hash"][:12])}</span></div>')
        if st.button("New note", icon=":material/add:", use_container_width=True):
            st.session_state.note = None
            st.rerun()
    if len(actionable):
        st.progress(done / len(actionable))
    kpis(r)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        for tier in ("critical", "review"):
            fs = [(i, f) for i, f in enumerate(r["findings"]) if f["tier"] == tier]
            if not fs:
                continue
            st.html(f'<div class="sec-title">{icon(TIERS[tier]["icon"], 17)}{TIERS[tier]["label"]}'
                    f'<span class="sec-count">{len(fs)}</span></div>')
            for i, f in fs:
                finding_card(f, i, r, decisions)
        info = [(i, f) for i, f in enumerate(r["findings"]) if f["tier"] == "info"]
        if not actionable:
            st.html(f'<div class="empty">{icon("circle-check", 22)}<div style="margin-top:8px">'
                    "No critical or review findings. Info items are listed below.</div></div>")
        if info:
            with st.expander(f"Info — {len(info)} lower-priority items (shown, never hidden)"):
                for i, f in info:
                    finding_card(f, i, r, decisions)
    with right:
        t1, t2, t3 = st.tabs(["Medications", "Note", "Run details"])
        with t1:
            medication_table(r)
        with t2:
            note_view(r, text)
        with t3:
            run_details(r)


# ---------------------------------------------------------------- audit page

def audit_page():
    sidebar()
    log = audit()
    ok, bad = log.verify()
    entries = log.entries()
    st.html(f"""<div class="page-head"><div><div class="eyebrow">Accountability</div>
      <div class="page-title">Audit log</div>
      <div class="page-sub">Every confirm and override, append-only and hash-chained. Rows cannot be
      edited or deleted; any tampering breaks the chain.</div></div></div>""")
    if ok:
        st.html(f'<div class="callout">{icon("shield-check", 16)}<div><b>Chain intact.</b> '
                f'{len(entries)} {"entry" if len(entries) == 1 else "entries"} verified from the first row.</div></div>')
    else:
        st.html(f'<div class="callout warn">{icon("triangle-alert", 16)}<div><b>Chain broken at row {bad}.</b> '
                f'The log file was modified outside Athena.</div></div>')
    if not entries:
        st.html('<div class="empty">No decisions recorded yet.</div>')
        return
    df = pd.DataFrame([{
        "time (UTC)": e.ts.replace("T", " ").replace("+00:00", ""), "reviewer": e.reviewer, "case": e.case_id,
        "tier": e.tier, "finding": e.title, "action": e.action,
        "reason": OVERRIDE_REASONS.get(e.reason_code, "") if e.reason_code else "", "comment": e.comment or "",
        "report": e.report_hash[:12], "row hash": e.row_hash[:12],
    } for e in reversed(entries)])
    cases = ["All cases"] + sorted(df["case"].unique())
    pick = st.selectbox("Case", cases)
    view = df if pick == "All cases" else df[df["case"] == pick]
    st.dataframe(view, hide_index=True, use_container_width=True,
                 column_config={"finding": st.column_config.TextColumn(width="large")})
    st.download_button("Export CSV", view.to_csv(index=False).encode(), "athena_audit.csv", "text/csv",
                       icon=":material/download:")


# ---------------------------------------------------------------- about page

def about_page():
    sidebar()
    st.html(f"""<div class="page-head"><div><div class="eyebrow">How it works</div>
      <div class="page-title">Neuro-symbolic medication safety</div>
      <div class="page-sub">A local language model reads; verified databases decide; a pharmacist confirms.</div>
      </div></div>
      <div class="kpis" style="grid-template-columns:repeat(3,minmax(0,1fr))">
        <div class="kpi k-accent"><div class="kpi-label">{icon("file-text", 14)}Branch A · Extraction</div>
          <div class="kpi-foot" style="margin-top:8px;color:var(--ink-2)">Llama 3.2 3B (4-bit, Ollama) plus deterministic
          rules read drug, dose, route and frequency. Every value must be found verbatim in the note — anything the
          model invents is discarded.</div></div>
        <div class="kpi k-accent"><div class="kpi-label">{icon("database", 14)}Branch B · Verification</div>
          <div class="kpi-foot" style="margin-top:8px;color:var(--ink-2)">Every active pair is checked against
          DrugBank and DDInter. "Not covered" is never reported as "safe".</div></div>
        <div class="kpi k-accent"><div class="kpi-label">{icon("shield-check", 14)}Fusion · Human review</div>
          <div class="kpi-foot" style="margin-top:8px;color:var(--ink-2)">Risk = severity × evidence × extraction
          confidence. Major interactions are never hidden. Each decision is logged.</div></div>
      </div>
      <div class="callout">{icon("lock", 16)}<div>Runs entirely on this laptop (8 GB, no GPU). No patient text is sent anywhere.</div></div>
      <div class="callout warn">{icon("triangle-alert", 16)}<div>Research prototype for decision support. Not a medical
      device; not validated for clinical use. See docs/limitations.md.</div></div>""")


pg = st.navigation([
    st.Page(review_page, title="Review", icon=":material/fact_check:", default=True),
    st.Page(audit_page, title="Audit log", icon=":material/history:", url_path="audit"),
    st.Page(about_page, title="About", icon=":material/info:", url_path="about"),
])
pg.run()
