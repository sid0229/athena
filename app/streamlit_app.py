"""Athena — pharmacist review dashboard.

    streamlit run app/streamlit_app.py

Runs fully on this machine: local LLM via Ollama on localhost, local SQLite KB and
audit log, fonts served from app/static. No patient text leaves the device.

This file is presentation only: it calls the existing pipeline (`analyze`), checker
and audit log unchanged and renders their results.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import datetime
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
from ui.theme import (TIERS, TYPE_LABELS, avatar, chip, css, esc, icon, label, meter,  # noqa: E402
                      section_title, tier_badge)

st.set_page_config(page_title="Athena · Medication Safety", page_icon=":material/medication:",
                   layout="wide", initial_sidebar_state="expanded")
st.html(css())
st.logo(str(Path(__file__).parent / "static" / "athena-logo.svg"), size="large")

DEMO_DIR = ROOT / "data" / "demo"
MODES = {"hybrid": "Hybrid — local LLM + rules", "rules": "Rules only — fast, no LLM"}
ROLES = ["Clinical reviewer", "Clinical pharmacist", "Physician", "Student reviewer"]


# ---------------------------------------------------------------- cached resources (unchanged)

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


# ---------------------------------------------------------------- presentation helpers

def _pretty_title(stem: str) -> str:
    """'af_pneumonia' -> 'AF Pneumonia' (short words read as abbreviations)."""
    words = stem.replace("_", " ").split()
    return " ".join(w.upper() if len(w) <= 3 else w.capitalize() for w in words)


def demo_notes() -> dict[str, str]:
    out = {}
    for p in sorted(DEMO_DIR.glob("*.txt")):
        out[_pretty_title(p.stem.split("_", 1)[-1])] = p.read_text()
    return out


def _reviewer() -> str:
    return (st.session_state.get("reviewer") or "").strip()


def _role() -> str:
    return st.session_state.get("reviewer_role") or ROLES[0]


def _ts(iso: str) -> str:
    return iso.replace("T", " ").replace("+00:00", "") + " UTC"


def _cap(name: str) -> str:
    return name[:1].upper() + name[1:] if name else name


def _pair_title(title: str) -> str:
    """'Major interaction: warfarin + acetylsalicylic acid' -> 'Warfarin + Acetylsalicylic acid'."""
    rest = title.split(": ", 1)[-1]
    return " + ".join(_cap(p) for p in rest.split(" + "))


def _note_stats(text: str) -> str:
    lines = text.count("\n") + 1
    return f"{lines:,} lines · {len(text):,} characters"


def _set_note(text: str, label_: str, source: str):
    st.session_state.note = text
    st.session_state.note_label = label_
    st.session_state.note_source = source
    st.session_state.analyzed_at = datetime.now().strftime("%H:%M")
    st.rerun()


def header(eyebrow: str, title: str, sub: str = "", meta_html: str = ""):
    """Page header with the reviewer context on the right."""
    left, right = st.columns([4, 1.3], vertical_alignment="top")
    with left:
        st.html(f'<div class="eyebrow">{esc(eyebrow)}</div><div class="page-title">{esc(title)}</div>'
                + (f'<div class="page-sub">{esc(sub)}</div>' if sub else "") + meta_html)
    with right:
        name = _reviewer()
        if name:
            st.html(f'<div class="who"><div class="account" role="group" aria-label="Current reviewer">{avatar(name)}'
                    f'<div><div class="who-label">Reviewer</div><div class="who-name">{esc(name)}</div>'
                    f'<div class="who-role">{esc(_role().title())}</div></div></div></div>')
        else:
            st.html(f'<div class="who"><div class="account account-empty">{icon("user", 16)}'
                    f'<div><div class="who-label">Reviewer</div><div class="who-none">Not set — use the sidebar</div>'
                    f'</div></div></div>')


# ---------------------------------------------------------------- sidebar

def _reviewer_form():
    with st.form("reviewer-form", border=False):
        name = st.text_input("Name", value=_reviewer(), placeholder="e.g. Dr. A. Sharma")
        role = st.selectbox("Role", ROLES, index=ROLES.index(_role()) if _role() in ROLES else 0)
        st.caption("Local session only — not authenticated. The name is written to the audit log.")
        if st.form_submit_button("Save reviewer", type="primary", width="stretch"):
            st.session_state.reviewer = name.strip()
            st.session_state.reviewer_role = role
            st.rerun()


def sidebar():
    with st.sidebar:
        st.html('<div class="side-label">Current reviewer</div>')
        name = _reviewer()
        if name:
            st.html(f"""<div class="id-card">{avatar(name)}
              <div><div class="id-name">{esc(name)}</div><div class="id-role">{esc(_role())}</div>
              <div class="id-state"><span class="dot dot-ok"></span>Active session</div></div></div>""")
        else:
            st.html(f"""<div class="id-card id-empty"><span class="avatar avatar-md avatar-empty">{icon("user", 18)}</span>
              <div><div class="id-name">No reviewer set</div>
              <div class="id-role">Required to confirm or override findings</div></div></div>""")
        with st.popover("Switch reviewer" if name else "Set reviewer", icon=":material/person:", width="stretch"):
            _reviewer_form()

        st.html('<div class="side-div"></div><div class="side-label">Extraction</div>')
        st.selectbox("Extraction mode", list(MODES), format_func=MODES.get, key="mode",
                     label_visibility="collapsed")

        online = llm_online()
        meta = kb_meta()
        model = extractor("hybrid").llm.model
        st.html(f"""
        <div class="side-div"></div><div class="side-label">System</div>
        <div class="status"><span class="dot {'dot-ok' if online else 'dot-off'}"></span><div>
          <div class="status-title">Local LLM</div>
          <div class="status-sub">{esc(model)} · {'online' if online else 'offline — rules fallback'}</div></div></div>
        <div class="status"><span class="dot dot-ok"></span><div>
          <div class="status-title">Knowledge base</div>
          <div class="status-sub">{int(meta['concepts']):,} drugs · {int(meta['pairs']):,} pairs</div></div></div>
        <div class="status"><span class="dot dot-ok"></span><div>
          <div class="status-title">Privacy</div>
          <div class="status-sub">On-device · no network calls</div></div></div>
        <div class="side-foot">Decision support only. Interaction flags come from DrugBank and DDInter
        records — never from the language model. A reviewer confirms every finding.</div>""")


# ---------------------------------------------------------------- input workspace

def _action_bar(text: str | None, button_label: str, key: str, source: str, label_: str):
    mode = st.session_state.get("mode", "hybrid")
    left, right = st.columns([3, 1.25], vertical_alignment="center")
    with left:
        stats = _note_stats(text) if text else "No note yet"
        st.html(f'<div class="action-meta"><span>{icon("file-text", 14)} {esc(stats)}</span>'
                f'<span>{icon("lock", 14)} Local analysis · no network calls</span>'
                f'<span>{icon("cpu", 14)} {esc(MODES[mode])}</span></div>')
    with right:
        if st.button(button_label, type="primary", key=key, disabled=not (text and text.strip()),
                     icon=":material/arrow_forward:", icon_position="right", width="stretch"):
            _set_note(text, label_, source)


def note_input():
    header("Medication reconciliation", "Review a discharge summary",
           "Extract medications, reconcile admission vs discharge, and verify active medication pairs "
           "against trusted interaction data.")
    t1, t2, t3 = st.tabs(["Demo patients", "Paste note", "Upload .txt"])
    with t1:
        notes = demo_notes()
        with st.container(key="panel-demo"):
            c1, c2 = st.columns([1, 1.6], gap="large")
            with c1:
                st.html(label("Patient"))
                choice = st.selectbox("Synthetic demo patient", list(notes), label_visibility="collapsed")
                st.html(f'<div class="ctx-name" style="margin-top:12px">{esc(choice)}</div>'
                        f'<div class="ctx-sub">Synthetic demo patient · fictional</div>')
                st.html(f'<div class="notice" style="margin-top:16px">{icon("info", 16)}<div><b>Demo environment</b>'
                        "These notes are fictional and written for this prototype. Real clinical notes must "
                        "remain on the authorized device.</div></div>")
            with c2:
                st.html(label("Input") + f'<div class="doc doc-preview" tabindex="0" '
                        f'aria-label="Selected demo note">{esc(notes[choice])}</div>')
            st.html('<div class="rule" style="margin:16px 0 12px"></div>')
            _action_bar(notes[choice], "Analyze discharge summary", "go-demo", "Synthetic demo patient", choice)
    with t2:
        with st.container(key="panel-paste"):
            st.html(label("Input"))
            pasted = st.text_area("Clinical note", height=320, placeholder="Paste a discharge summary…",
                                  label_visibility="collapsed")
            st.html('<div class="rule" style="margin:12px 0"></div>')
            _action_bar(pasted, "Analyze note", "go-paste", "Pasted note", "Pasted discharge summary")
    with t3:
        with st.container(key="panel-upload"):
            st.html(label("Input"))
            up = st.file_uploader("Upload a plain-text note", type=["txt"], label_visibility="collapsed")
            text = up.getvalue().decode("utf-8", errors="replace") if up is not None else None
            if text:
                st.html(f'<div class="doc doc-preview" tabindex="0" aria-label="Uploaded note">{esc(text)}</div>')
            st.html('<div class="rule" style="margin:12px 0"></div>')
            _action_bar(text, "Analyze uploaded note", "go-up", "Uploaded file",
                        up.name if up is not None else "Uploaded note")


# ---------------------------------------------------------------- results

def kpis(r: dict):
    c = r["meta"]["counts"]
    tiers = c["tiers"]
    tiles = [
        ("k-critical", TIERS["critical"]["icon"], "Critical", tiers["critical"], "act before discharge"),
        ("k-review", TIERS["review"]["icon"], "Review", tiers["review"], "reviewer judgement"),
        ("k-info", TIERS["info"]["icon"], "Info", tiers["info"], "shown, lower priority"),
        ("", "pill", "Active meds", c["active"], f"of {c['medications']} medications in note"),
    ]
    act = tiers["critical"] + tiers["review"]
    st.html('<div class="kpis" role="status" aria-label="' + f'{act} findings need review">' + "".join(
        f'<div class="kpi {k}{" k-zero" if v == 0 else ""}"><div class="kpi-label">{icon(ic, 14)}{esc(lbl)}</div>'
        f'<div class="kpi-value">{v:,}</div><div class="kpi-foot">{esc(foot)}</div></div>'
        for k, ic, lbl, v, foot in tiles) + "</div>")


def _med_conf(r: dict, ids: list[str]) -> float | None:
    conf = {m["id"]: m["confidence"] for m in r["medications"]}
    vals = [conf[i] for i in ids if i in conf]
    return min(vals) if vals else None


def finding_card(f: dict, idx: int, r: dict, decisions: dict):
    """Collapsed by default: what / why / how serious / action. Evidence on demand."""
    tier = f["tier"]
    tlabel, ticon = TYPE_LABELS.get(f["type"], (f["type"], "info"))
    ev = f.get("evidence") or {}
    is_int = f["type"] == "INTERACTION"
    expl = list(f["explanation"])
    risk_lines = [x for x in expl if x.startswith("Risk ")]
    reasoning = [x for x in expl if not x.startswith("Risk ")]

    with st.container(key=f"card-{tier}-{idx}"):
        # ---- primary content (always visible)
        sev = ""
        if is_int and ev.get("severity"):
            derived = '<small>derived</small>' if ev.get("severity_basis") == "derived" else ""
            sev = f'<span class="f-sev sev-{ev["severity"].lower()}">{esc(ev["severity"])}{derived}</span>'
        title = _pair_title(f["title"]) if is_int else _cap(f["title"].split(": ", 1)[-1])
        if is_int and ev.get("mechanisms"):
            summary = ev["mechanisms"][0]["text"]
        elif is_int and ev.get("ddinter_level"):
            summary = f"DDInter grades this pair {ev['ddinter_level']}."
        else:
            summary = reasoning[0] if reasoning else ""
        verify = (f'<div class="verify">{icon("triangle-alert", 14)} Possible Major interaction — verify the '
                  f'extracted drugs</div>') if f.get("needs_verification") else ""
        stats = []
        if is_int:
            stats.append(f"Risk <b>{f['risk']:.2f}</b>")
        conf = _med_conf(r, f["meds"])
        if conf is not None and f["type"] not in ("NOT_CHECKED", "NOT_COVERED"):
            stats.append(f"Confidence <b>{conf:.2f}</b>")
        if ev.get("sources"):
            n = len(ev["sources"])
            stats.append(f"{n} source{'s' if n != 1 else ''}")
        d = decisions.get(f["id"])
        history = ""
        if d:
            what = ("Confirmed" if d.action == "confirm"
                    else OVERRIDE_REASONS.get(d.reason_code, "Overridden"))
            history = (f'<div class="f-history">{icon("circle-check" if d.action == "confirm" else "undo", 13)}'
                       f'Previously reviewed · {esc(d.reviewer)} · {esc(what)}'
                       f'{(" — " + esc(d.comment)) if d.comment else ""}'
                       f'<span class="f-history-ts">{esc(_ts(d.ts))}</span></div>')
        st.html(f"""
        <div class="f-head">{tier_badge(tier)}<span class="f-type">{icon(ticon, 14)}{esc(tlabel)}</span>{sev}</div>
        <div class="f-title">{esc(title)}</div>
        {('<div class="f-summary">' + esc(summary) + '</div>') if summary else ''}{verify}
        {('<div class="f-stats">' + '<span class="sep">·</span>'.join(f'<span>{x}</span>' for x in stats) + '</div>') if stats else ''}
        {history}""")

        # ---- actions
        if f["type"] not in ("NOT_CHECKED", "NOT_COVERED"):
            reviewer = _reviewer()
            hint = None if reviewer else "Set a reviewer in the sidebar first"
            open_key = f"ovopen-{f['id']}"  # display-only: is the inline override panel open?
            is_open = bool(st.session_state.get(open_key)) and bool(reviewer)
            c1, c2, _ = st.columns([1, 1.15, 1.9], vertical_alignment="center")
            with c1:
                if st.button("Confirm", key=f"ok-{idx}", type="primary", icon=":material/check:",
                             disabled=not reviewer, help=hint, width="stretch"):
                    st.session_state[open_key] = False
                    _record(r, f, "confirm")
            with c2:
                st.button("Override", key=f"ovt-{idx}", disabled=not reviewer, width="stretch",
                          icon=":material/expand_less:" if is_open else ":material/expand_more:",
                          icon_position="right", on_click=_toggle, args=(open_key,),
                          help=hint or ("Close override" if is_open else "Override this finding — a reason is required"))
            if is_open:
                with st.container(key=f"ovpanel-{idx}"):
                    st.html(f'<div class="ov-title">{icon("edit-note", 15)}Override decision</div>'
                            f'<div class="ov-sub">You are overriding: <b>{esc(title)}</b></div>')
                    reason = st.selectbox("Reason *", list(OVERRIDE_REASONS),
                                          format_func=OVERRIDE_REASONS.get, key=f"rs-{idx}")
                    note = st.text_area("Comment *" if reason == "other" else "Comment", key=f"cm-{idx}",
                                        height=84, placeholder=("Required when the reason is “Other”"
                                                                if reason == "other" else "Optional comment…"))
                    h1, h2 = st.columns([2.2, 1], vertical_alignment="center")
                    with h1:
                        st.html('<div class="ov-help">Override decisions are recorded in the audit log '
                                'with your name and the reason.</div>')
                    with h2:
                        if st.button("Record override", key=f"ov-{idx}", type="primary", width="stretch",
                                     disabled=reason == "other" and not note.strip()):
                            st.session_state[open_key] = False
                            _record(r, f, "override", reason, note)

        # ---- evidence & reasoning (collapsed)
        with st.expander("View evidence & reasoning"):
            blocks = []
            if ev.get("sources") or ev.get("mechanisms") or ev.get("ddinter_level"):
                chips = "".join(chip("DDInter" if s == "ddinter" else "DrugBank", "accent", "database")
                                for s in ev.get("sources", []))
                if ev.get("severity_basis") == "derived":
                    chips += chip("Severity derived", "warn")
                lines = []
                if ev.get("ddinter_level"):
                    lines.append(f"<li>DDInter grade: <b>{esc(ev['ddinter_level'])}</b></li>")
                for m in ev.get("mechanisms", []):
                    lines.append(f"<li>DrugBank: {esc(m['text'])}"
                                 + (f' <span class="ev-sub">(template: {esc(m["derived_severity"])}, '
                                    f'{m["pct_major"]:.0f}% Major of {m["n_overlap_graded"]} comparable pairs)</span>'
                                    if ev.get("severity_basis") == "derived" else "") + "</li>")
                blocks.append(("Evidence", f'<div>{chips}</div>'
                               + (f'<ul class="ev-list">{"".join(lines)}</ul>' if lines else "")))
            if reasoning:
                blocks.append(("Reasoning", '<ul class="ev-list">' + "".join(f"<li>{esc(x)}</li>" for x in reasoning)
                               + "</ul>"))
            conf_parts = (ev.get("confidence") or {})
            if conf_parts:
                names = {m["id"]: _cap(m["name"]) for m in r["medications"]}
                totals = {m["id"]: m["confidence"] for m in r["medications"]}
                rows = "".join(
                    f'<tr><td>{esc(names.get(mid, mid))}</td><td><b>{totals.get(mid, 0):.2f}</b></td>'
                    f'<td>{" · ".join(f"{k} {v:.2f}" for k, v in parts.items())}</td></tr>'
                    for mid, parts in conf_parts.items())
                blocks.append(("Confidence", f'<table class="ev-table"><tbody>{rows}</tbody></table>'))
            elif conf is not None:
                blocks.append(("Confidence", f'<div class="ev-text">Extraction confidence {conf:.2f}</div>'))
            if risk_lines:
                blocks.append(("Risk calculation", "".join(f'<div class="ev-text">{esc(x)}</div>' for x in risk_lines)))
            st.html('<div class="ev-grid">' + "".join(
                f'<div class="ev-block"><div class="f-label">{esc(t)}</div>{body}</div>' for t, body in blocks)
                + "</div>")


def _toggle(key: str):
    st.session_state[key] = not st.session_state.get(key, False)


def _record(r: dict, f: dict, action: str, reason: str | None = None, comment: str | None = None):
    audit().record(reviewer=st.session_state.reviewer, case_id=case_id_for(st.session_state.note),
                   report_hash=r["meta"]["report_hash"], finding_id=f["id"], finding_type=f["type"],
                   tier=f["tier"], title=f["title"], action=action, reason_code=reason, comment=comment)
    st.toast(f"{'Confirmed' if action == 'confirm' else 'Override recorded'} · audit log updated",
             icon=":material/verified:")
    st.rerun()


PRIOR_LIST_HEADER = "Medications on Admission:"


def _with_prior_list(note: str, prior: str) -> str:
    """Attach a prior medication list as an admission-medications section so the existing
    pipeline reconciles it (the same path as a note that already contains both lists).
    A neutral header after the list ends the section before the original note begins."""
    return f"{PRIOR_LIST_HEADER}\n{prior.strip()}\n\nDischarge Summary:\n{note}"


def reconciliation_setup(r: dict, text: str):
    """Right panel when the note has a discharge list but no prior (admission) list."""
    meds = r["medications"]
    n_dis = sum(m["in_discharge"] for m in meds)
    n_active = r["meta"]["counts"]["active"]
    with st.container(key="panel-recon"):
        st.html(f"""<div class="field-label">Discharge review</div>
          <div class="rs-count"><span class="rs-num">{n_dis}</span> discharge medication{'s' if n_dis != 1 else ''}</div>
          <div class="rs-ok">{icon("circle-check", 15)} Extracted and checked for interactions ·
            {n_active} active</div>
          <div class="rs-div"></div>
          <div class="field-label rs-label">{icon("info", 14)} Reconciliation</div>
          <div class="rs-text">Compare these medications with a prior medication list to find omissions,
            new medications and dose changes.</div>""")
        open_key = "prior-list-open"
        if not st.session_state.get(open_key):
            st.button("Add prior medication list", key="prior-open", type="primary", icon=":material/add:",
                      on_click=_toggle, args=(open_key,))
            st.html('<div class="rs-muted">No prior medication list was provided.</div>')
        else:
            prior = st.text_area("Prior medication list", key="prior-list", height=150,
                                 placeholder="One medication per line, e.g.\nMetoprolol succinate 50 mg PO daily\n"
                                             "Lisinopril 20 mg PO daily")
            st.html('<div class="rs-muted">Athena re-analyzes this discharge summary together with the list. '
                    'This creates a new analysis; everything stays on this device.</div>')
            b1, b2 = st.columns([1.4, 1])
            with b1:
                if st.button("Reconcile with this list", key="prior-go", type="primary", width="stretch",
                             icon=":material/compare_arrows:", disabled=not prior.strip()):
                    st.session_state[open_key] = False
                    base = st.session_state.get("note_label") or "Discharge summary"
                    _set_note(_with_prior_list(text, prior), f"{base} + prior list",
                              "Discharge summary + prior medication list")
            with b2:
                st.button("Cancel", key="prior-cancel", width="stretch", on_click=_toggle, args=(open_key,))


def reconciliation(r: dict, text: str):
    kinds = {s["kind"] for s in r["meta"]["sections"]}
    if "discharge" in kinds and "admission" not in kinds:
        reconciliation_setup(r, text)
        return
    st.html(section_title("Admission → Discharge"))
    if not {"admission", "discharge"} <= kinds:
        # No discharge medication list at all: informational, not an error.
        st.html(f"""<div class="rs-card">{icon("info", 16)}<div>
          <div class="rs-title">No discharge medication list found</div>
          <div class="rs-text">Athena treated every medication mentioned in the note as active and checked
          them for interactions. Reconciliation needs a discharge medication list.</div></div></div>""")
        return
    by_id = {f["id"]: f for f in r["findings"]}
    arrow = icon("arrow-right", 14)
    rows = []
    for m in r["medications"]:
        if not (m["in_admission"] or m["in_discharge"]):
            continue
        adm_s = m["admission_attributes"].get("Strength") or ("—" if not m["in_admission"] else "dose not stated")
        dis_s = (m["attributes"].get("Strength") or "dose not stated") if m["in_discharge"] else None
        name = _cap(m["name"])
        if m["in_admission"] and not m["in_discharge"]:
            f = by_id.get(f"OMIT:{m['id']}")
            explained = f is not None and f["tier"] == "info"
            order = 1 if explained else 0
            dose = f'{esc(adm_s)} {arrow} <span class="rc-gone">Not on discharge list</span>'
            badge = (f'<span class="badge tier-info">{icon("info", 12)}Stopped / held</span>' if explained else
                     f'<span class="badge tier-review">{icon("triangle-alert", 12)}Omission</span>')
        elif m["in_discharge"] and not m["in_admission"]:
            order = 3
            dose = f'— {arrow} {esc(dis_s)}'
            badge = f'<span class="badge tier-info">{icon("plus", 12)}New medication</span>'
        else:
            changed = f"DOSE:{m['id']}" in by_id
            order = 2 if changed else 4
            dose = f'{esc(adm_s)} {arrow} {esc(dis_s)}' if changed else esc(dis_s)
            badge = (f'<span class="badge tier-review">{icon("repeat", 12)}Dose change</span>' if changed
                     else f'<span class="rc-same">{icon("check", 13, 2.4)}Unchanged</span>')
        rows.append((order, name.lower(), f'<div class="rc-row"><div class="rc-main"><div class="rc-name">{esc(name)}</div>'
                     f'<div class="rc-dose">{dose}</div></div><div class="rc-badge">{badge}</div></div>'))
    rows.sort()
    n = {k: sum(1 for o, *_ in rows if o == k) for k in range(5)}
    foot = (f"{n[0]} omission{'s' if n[0] != 1 else ''} · {n[2]} dose change{'s' if n[2] != 1 else ''} · "
            f"{n[3]} new · {n[1]} stopped/held · {n[4]} unchanged")
    st.html('<div class="recon"><div class="rc-head"><span>Medication · admission → discharge</span><span>Change</span></div>'
            + "".join(x for *_, x in rows) + f'<div class="recon-foot">{esc(foot)}</div></div>')


def medication_table(r: dict):
    rows = []
    for m in r["medications"]:
        a = m["attributes"]
        dose = " · ".join(x for x in (a.get("Strength"), a.get("Route"), a.get("Frequency")) if x)
        status_ic = {"active": "circle-check", "held": "undo", "stopped": "x", "mentioned": "info"}[m["status"]]
        status = f'<span class="st-pill st-{m["status"]}">{icon(status_ic, 14)}{esc(m["status"].title())}</span>'
        if m["text_status"]:
            status += f'<div class="med-raw">note says “{esc(m["text_status"])}”</div>'
        yes = f'<span class="yes" aria-label="yes">{icon("check", 16, 2.6)}</span>'
        no = '<span class="no" aria-label="no">—</span>'
        rows.append(f"""<tr>
          <td><div class="med-name">{esc(_cap(m['name']))}</div>
              <div class="med-raw">as written: “{esc(m['raw'])}” · {esc(m['norm_status'].replace('_', ' '))}</div>
              {('<div class="med-dose">' + esc(dose) + '</div>') if dose else ''}</td>
          <td>{status}</td>
          <td class="c">{yes if m['in_admission'] else no}</td>
          <td class="c">{yes if m['in_discharge'] else no}</td>
          <td>{meter(m['confidence'], 'extraction confidence')}</td></tr>""")
    st.html(f"""<table class="medtab" aria-label="Extracted medications">
      <thead><tr><th>Medication</th><th>Status</th><th style="text-align:center">Admission</th>
      <th style="text-align:center">Discharge</th><th>Confidence</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>""")


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
    st.html('<div class="legend"><span><mark class="d">drug</mark> extracted</span>'
            '<span><mark class="d hot">drug</mark> in a critical finding</span></div>'
            f'<div class="doc doc-full" tabindex="0" aria-label="Discharge summary with extracted drugs highlighted">'
            f'<div class="doc-head">Discharge summary</div>{"".join(out)}</div>')


def run_details(r: dict, cid: str = ""):
    m = r["meta"]
    secs = ", ".join(f"{s['kind']} ({s['title']})" for s in m["sections"]) or "none found"
    for w in m["warnings"]:
        st.html(f'<div class="notice warn" style="margin-bottom:12px">{icon("triangle-alert", 16)}<div>{esc(w)}</div></div>')
    c = m["counts"]
    st.html(f"""<table class="kv"><tbody>
      <tr><td>Extraction mode</td><td>{esc(m['extraction_mode'])}</td></tr>
      <tr><td>Local model</td><td>{esc(m.get('llm_model') or '—')}</td></tr>
      <tr><td>Processing time</td><td>{m['extraction_seconds']} s</td></tr>
      <tr><td>Medication sections</td><td>{esc(secs)}</td></tr>
      <tr><td>Pairs checked</td><td>{c['pairs_checked']} · {c['interactions']} with a record ·
          {c['no_known_interaction']} no known interaction · {c['not_covered']} not covered</td></tr>
      <tr><td>Case ID</td><td class="hash">{esc(cid)}</td></tr>
      <tr><td>Report hash</td><td class="hash" style="word-break:break-all">{esc(m['report_hash'])}</td></tr>
    </tbody></table>""")


def _apply_query_params():
    """Shortcuts for demos: ?demo=1..N loads a synthetic note, ?mode=rules|hybrid, ?reviewer=Name."""
    qp = st.query_params
    if "mode" in qp and qp["mode"] in MODES:
        st.session_state.mode = qp["mode"]
    if "reviewer" in qp and not st.session_state.get("reviewer"):
        st.session_state.reviewer = qp["reviewer"]
    if "demo" in qp and not st.session_state.get("_demo_loaded"):
        notes = demo_notes()
        i = int(qp["demo"]) - 1 if qp["demo"].isdigit() else 0
        if 0 <= i < len(notes):
            label_, text = list(notes.items())[i]
            st.session_state.note = text
            st.session_state.note_label = label_
            st.session_state.note_source = "Synthetic demo patient"
            st.session_state.analyzed_at = datetime.now().strftime("%H:%M")
            st.session_state._demo_loaded = True


def review_page():
    _apply_query_params()
    sidebar()
    text = st.session_state.get("note")
    if not text:
        note_input()
        return
    mode = st.session_state.get("mode", "hybrid")
    with st.spinner("Reading the note on this device and checking every medication pair…"):
        r = run_analysis(text, mode)
    cid = case_id_for(text)
    decisions = audit().latest_decisions(cid, r["meta"]["report_hash"])
    actionable = [f for f in r["findings"] if f["tier"] in ("critical", "review")]
    done = sum(f["id"] in decisions for f in actionable)

    label_ = st.session_state.get("note_label") or cid
    source = st.session_state.get("note_source") or "Discharge summary"
    when = st.session_state.get("analyzed_at")
    meta = (f'<div class="meta-line"><span>{icon("file-text", 14)} {esc(source)}</span>'
            + (f'<span>{icon("clock", 14)} Analyzed {esc(when)}</span>' if when else "")
            + f'<span>{icon("cpu", 14)} {esc(MODES[mode])}</span></div>')
    header("Medication reconciliation", label_, meta_html=meta)

    bar_l, bar_r = st.columns([4, 1.3], vertical_alignment="center")
    with bar_l:
        total = len(actionable)
        pct = round(100 * done / total) if total else 100
        st.html(f'<div class="progress-wf" role="progressbar" aria-valuenow="{done}" aria-valuemin="0" '
                f'aria-valuemax="{total}" aria-label="{done} of {total} actionable findings reviewed">'
                f'<div class="pw-text"><span class="pw-num">{done} of {total}</span> reviewed'
                f'<span class="pw-help">actionable findings · Critical and Review</span></div>'
                f'<div class="pw-track"><div class="pw-fill" style="width:{pct}%"></div></div></div>')
    with bar_r:
        if st.button("New note", icon=":material/add:", width="stretch"):
            st.session_state.note = None
            st.rerun()
    st.html('<div class="gap-4"></div>')
    kpis(r)
    st.html('<div class="gap-6"></div>')

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.html(section_title("Needs review", len(actionable)))
        if not actionable:
            st.html(f'<div class="empty">{icon("circle-check", 22)}<div style="margin-top:8px">'
                    "No critical or review findings. Lower-priority items are listed below.</div></div>")
        for tier in ("critical", "review"):
            for i, f in enumerate(r["findings"]):
                if f["tier"] == tier:
                    finding_card(f, i, r, decisions)
        info = [(i, f) for i, f in enumerate(r["findings"]) if f["tier"] == "info"]
        if info:
            with st.expander(f"Info · {len(info)} lower-priority items (shown, never hidden)"):
                for i, f in info:
                    finding_card(f, i, r, decisions)
    with right:
        reconciliation(r, text)

    st.html('<div class="rule"></div>' + section_title("Medication record", sub="extracted list, source note and run details"))
    t1, t2, t3 = st.tabs([f"Medications ({len(r['medications'])})", "Source note", "Run details"])
    with t1:
        medication_table(r)
    with t2:
        note_view(r, text)
    with t3:
        run_details(r, cid)


# ---------------------------------------------------------------- audit page

def audit_page():
    sidebar()
    log = audit()
    ok, bad = log.verify()
    entries = log.entries()
    header("Accountability", "Audit log",
           "Every confirm and override, append-only and hash-chained. Rows cannot be edited or deleted; "
           "any tampering breaks the chain.")
    if ok:
        st.html(f'<div class="notice ok">{icon("shield-check", 16)}<div><b>Chain intact</b>'
                f'{len(entries)} {"entry" if len(entries) == 1 else "entries"} verified from the first row.</div></div>')
    else:
        st.html(f'<div class="notice bad">{icon("triangle-alert", 16)}<div><b>Chain broken at row {bad}</b>'
                "The log file was modified outside Athena.</div></div>")
    if not entries:
        st.html('<div class="empty" style="margin-top:16px">No decisions recorded yet.</div>')
        return

    df = pd.DataFrame([{
        "time (UTC)": e.ts.replace("T", " ").replace("+00:00", ""), "reviewer": e.reviewer, "case": e.case_id,
        "tier": e.tier, "finding": e.title, "action": e.action,
        "reason": OVERRIDE_REASONS.get(e.reason_code, "") if e.reason_code else "", "comment": e.comment or "",
        "report": e.report_hash[:12], "row hash": e.row_hash[:12],
    } for e in reversed(entries)])
    st.html('<div style="height:16px"></div>')
    f1, f2 = st.columns([3, 1], vertical_alignment="bottom")
    with f1:
        cases = ["All cases"] + sorted(df["case"].unique())
        pick = st.selectbox("Case", cases)
    view = df if pick == "All cases" else df[df["case"] == pick]
    with f2:
        st.download_button("Export CSV", view.to_csv(index=False).encode(), "athena_audit.csv", "text/csv",
                           icon=":material/download:", width="stretch")

    shown = [e for e in reversed(entries) if pick == "All cases" or e.case_id == pick]
    rows, day = [], None
    for e in shown:
        d, t = e.ts[:10], e.ts[11:16]
        if d != day:
            rows.append(f'<div class="tl-day">{esc(d)} · UTC</div>')
            day = d
        verb = ("<span class='verb-confirm'>Confirmed</span>" if e.action == "confirm"
                else "<span class='verb-override'>Overrode</span>")
        reason = ""
        if e.reason_code:
            reason = f'<div class="tl-reason">Reason: {esc(OVERRIDE_REASONS.get(e.reason_code, e.reason_code))}' \
                     + (f" — {esc(e.comment)}" if e.comment else "") + "</div>"
        elif e.comment:
            reason = f'<div class="tl-reason">{esc(e.comment)}</div>'
        rows.append(f"""<div class="tl-row"><div class="tl-time">{esc(t)}<small>UTC</small></div>{avatar(e.reviewer, "sm")}
          <div><div class="tl-who">{esc(e.reviewer)} {tier_badge(e.tier)}</div>
          <div class="tl-act">{verb}: {esc(e.title)}</div>{reason}
          <div class="tl-meta">{esc(e.case_id)} · report {esc(e.report_hash[:12])} · row {esc(e.row_hash[:12])}</div></div></div>""")
    st.html(f'<div class="timeline" style="margin-top:8px">{"".join(rows)}</div>')
    with st.expander("Table view"):
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"finding": st.column_config.TextColumn(width="large")})


# ---------------------------------------------------------------- about page

def about_page():
    sidebar()
    header("How it works", "Neuro-symbolic medication safety",
           "A local language model reads; verified databases decide; a reviewer confirms.")
    st.html(f"""<div class="about-grid">
        <div class="about-card">{icon("file-text", 20)}<h4>Branch A · Extraction</h4>
          <p>Llama 3.2 3B (4-bit, Ollama) plus deterministic rules read drug, dose, route and frequency.
          Every value must be found verbatim in the note — anything the model invents is discarded.</p></div>
        <div class="about-card">{icon("database", 20)}<h4>Branch B · Verification</h4>
          <p>Every active pair is checked against DrugBank and DDInter. "Not covered" is never reported
          as "safe".</p></div>
        <div class="about-card">{icon("shield-check", 20)}<h4>Fusion · Human review</h4>
          <p>Risk = severity × evidence × extraction confidence. Major interactions are never hidden.
          Each decision is logged.</p></div>
      </div>
      <div class="notice" style="margin-bottom:12px">{icon("lock", 16)}<div><b>Privacy</b>Runs entirely on this
      laptop (8 GB, no GPU). No patient text is sent anywhere.</div></div>
      <div class="notice warn">{icon("triangle-alert", 16)}<div><b>Research prototype</b>Decision support only.
      Not a medical device; not validated for clinical use. See docs/limitations.md.</div></div>""")


pg = st.navigation([
    st.Page(review_page, title="Review", icon=":material/fact_check:", default=True),
    st.Page(audit_page, title="Audit log", icon=":material/history:", url_path="audit"),
    st.Page(about_page, title="About", icon=":material/info:", url_path="about"),
])
pg.run()
