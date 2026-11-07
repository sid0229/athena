"""Batch 5: sections, medication list, reconciliation, risk tiers, report hash.
Uses the rules extractor so no LLM is needed."""

import pytest

from athena.config import data_path
from athena.extraction.pipeline import MedicationExtractor
from athena.fusion.report import analyze
from athena.fusion.sections import medication_sections

NOTE = """Admission Date:  [**2115-2-22**]

Medications on Admission:
Coumadin 5 mg daily
Lasix 80 mg PO daily
Simvastatin 40 mg qhs
Ibuprofen prn

Brief Hospital Course:
Simvastatin was held given elevated LFTs. Coumadin was continued.

Discharge Medications:
1. warfarin 5 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).
2. aspirin 81 mg Tablet, Chewable Sig: One (1) Tablet, Chewable PO DAILY (Daily).
3. furosemide 40 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).
4. clarithromycin 500 mg Tablet Sig: One (1) Tablet PO BID (2 times a day) for 7 days.
5. acetaminophen 325 mg Tablet Sig: Two (2) Tablet PO Q6H (every 6
hours) as needed for pain.
6. Percocet 5-325 mg Tablet Sig: One (1) Tablet PO Q4H (every 4 hours) as needed for pain.
7. guaifenesin 100 mg/5 mL Syrup Sig: One (1) Tablet PO BID (2 times a day).

Discharge Disposition:
Home
"""


@pytest.fixture(scope="module")
def report():
    if not data_path("kb_db").exists():
        pytest.skip("run scripts/build_kb.py first")
    return analyze(NOTE, MedicationExtractor("rules"))


def by_type(report, t):
    return [f for f in report.findings if f.type == t]


def test_sections_found_and_sig_lines_do_not_end_them():
    secs = medication_sections(NOTE)
    assert [s.kind for s in secs] == ["admission", "discharge"]
    dis = secs[1]
    assert "guaifenesin" in NOTE[dis.start:dis.end]


def test_status_from_sections(report):
    st = {m.name: m.status for m in report.medications}
    assert st["warfarin"] == "active"
    assert st["clarithromycin"] == "active"
    assert st["simvastatin"] in ("held", "mentioned")  # not on discharge list
    assert st["ibuprofen"] == "mentioned"


def test_major_interactions_are_critical_with_evidence(report):
    ints = {f.title: f for f in by_type(report, "INTERACTION")}
    f = ints["Major interaction: warfarin + acetylsalicylic acid"]
    assert f.tier == "critical"
    assert f.evidence["ddinter_level"] == "Major" and "drugbank" in f.evidence["sources"]
    assert any("DDInter grades this pair: Major" in line for line in f.explanation)
    assert any(line.startswith("Risk ") for line in f.explanation)


def test_held_drug_is_not_paired(report):
    assert not any("simvastatin" in f.title for f in by_type(report, "INTERACTION"))


def test_duplicate_acetaminophen(report):
    assert [f.title for f in by_type(report, "DUPLICATE_THERAPY")] == ["Duplicate therapy: acetaminophen"]


def test_reconciliation(report):
    omit = {f.title: f for f in by_type(report, "OMISSION")}
    assert omit["Not on discharge list: ibuprofen"].tier == "review"
    assert any(t.endswith("simvastatin") for t in omit)
    assert omit[next(t for t in omit if t.endswith("simvastatin"))].tier == "info"  # explained: held
    doses = [f.title for f in by_type(report, "DOSE_CHANGE")]
    assert doses == ["Strength changed: furosemide 80 mg → 40 mg"]
    assert any(f.title == "New at discharge: clarithromycin" for f in by_type(report, "NEW_MEDICATION"))


def test_unchecked_medications_surface(report):
    nc = by_type(report, "NOT_CHECKED")
    assert nc and any("guaifenesin" in line for line in nc[0].explanation)


def test_major_never_below_review(report):
    for f in by_type(report, "INTERACTION"):
        if f.evidence["severity"] == "Major":
            assert f.tier in ("critical", "review")


def test_findings_sorted_and_hash_stable(report):
    order = {"critical": 0, "review": 1, "info": 2}
    tiers = [order[f.tier] for f in report.findings]
    assert tiers == sorted(tiers)
    again = analyze(NOTE, MedicationExtractor("rules"))
    assert again.report_hash == report.report_hash


def test_status_cue_attaches_to_nearest_preceding_drug():
    from athena.fusion.report import _rule_status

    t = "Simvastatin was held while on clarithromycin. We discontinued lisinopril."
    sp = {w: (t.index(w), t.index(w) + len(w)) for w in ("Simvastatin", "clarithromycin", "lisinopril")}
    status = {w: _rule_status(t, *s, [o for o in sp.values() if o != s]) for w, s in sp.items()}
    assert status == {"Simvastatin": "held", "clarithromycin": "", "lisinopril": "stopped"}


def test_local_route_detection():
    from athena.fusion.report import MedEntry, local_route

    def entry(raw, kb, **attrs):
        return MedEntry(id=raw, name=raw, raw=raw, norm_status="mapped", norm_stage="exact", kb_names=kb,
                        mentions=[], attributes=attrs, in_admission=False, in_discharge=True,
                        text_status="", status="active")
    assert local_route(entry("lidocaine", ["lidocaine"], Form="Patch")) == "topical"
    assert local_route(entry("fentanyl", ["fentanyl"], Form="Patch 72 hr")) is None  # transdermal = systemic
    assert local_route(entry("timolol", ["timolol"], Route="both eyes")) == "ophthalmic"
    assert local_route(entry("clobetasol", ["clobetasol"], Form="Cream")) == "topical"
    assert local_route(entry("metoprolol", ["metoprolol"], Route="PO", Form="Tablet")) is None


# ---- code review regressions (Nov 2026)

def test_colon_lines_inside_a_list_do_not_end_it():
    t = ("Discharge Medications:\n1. Warfarin 5 mg daily\nInsulin Lispro: sliding scale\n"
         "2. Metoprolol 25 mg BID\n\nDischarge Disposition:\nHome")
    (sec,) = medication_sections(t)
    assert "Metoprolol" in t[sec.start:sec.end] and "Disposition" not in t[sec.start:sec.end]


def test_list_ends_at_prose_and_header_variants_are_found():
    t = "Discharge Medications:\nwarfarin 5 mg daily\n\nAvoid aspirin and ibuprofen while on warfarin.\n"
    (sec,) = medication_sections(t)
    assert "aspirin" not in t[sec.start:sec.end]
    kinds = [s.kind for s in medication_sections(
        "Medications on Admission (home):\nmetoprolol 25 mg\n\nDischarge Medication:\nwarfarin 5 mg")]
    assert kinds == ["admission", "discharge"]
    assert [s.kind for s in medication_sections("DISCHARGE MEDICATIONS\nwarfarin 5 mg daily\n")] == ["discharge"]


def test_allergy_and_negated_mentions_are_not_active_without_discharge_list():
    r = analyze("Patient is allergic to aspirin. Takes warfarin 5 mg daily.", MedicationExtractor("rules"))
    status = {m.name: m.status for m in r.medications}
    assert status["acetylsalicylic acid"] == "mentioned" and status["warfarin"] == "active"
    assert not any(f.tier == "critical" for f in r.findings)


def test_status_cue_attaches_to_its_object_and_respects_negation():
    r = analyze("Warfarin was continued, held aspirin due to bleeding. Lisinopril was not held.",
                MedicationExtractor("rules"))
    status = {m.name: m.status for m in r.medications}
    assert status["acetylsalicylic acid"] == "held"
    assert status["warfarin"] == "active" and status["lisinopril"] == "active"
