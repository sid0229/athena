"""Batch 3: Branch B checker."""

import time

import pytest

from athena.config import data_path
from athena.verification.checker import InteractionChecker, Medication, PairStatus


@pytest.fixture(scope="module")
def checker():
    if not data_path("kb_db").exists():
        pytest.skip("run scripts/build_kb.py first")
    return InteractionChecker()


def find(report, x, y):
    for p in report.pairs:
        if {p.a.name, p.b.name} == {x, y}:
            return p
    raise AssertionError(f"pair {x}+{y} not in report")


def test_major_from_ddinter_via_brand_and_abbreviation(checker):
    r = checker.check(["Coumadin 5 mg", "ASA 81 mg"])
    p = find(r, "warfarin", "acetylsalicylic acid")
    assert p.status is PairStatus.INTERACTION
    assert (p.severity, p.severity_basis) == ("Major", "ddinter")
    assert set(p.sources) == {"ddinter", "drugbank"}


def test_mechanism_text_keeps_source_direction(checker):
    p = find(checker.check(["aspirin", "furosemide"]), "acetylsalicylic acid", "furosemide")
    assert any(m.text == "Acetylsalicylic acid may decrease the diuretic activities of Furosemide."
               for m in p.mechanisms)


def test_biologic_covered_by_ddinter(checker):
    p = find(checker.check(["heparin sodium", "warfarin"]), "heparin", "warfarin")
    assert (p.status, p.severity) == (PairStatus.INTERACTION, "Major")


def test_derived_severity_when_only_drugbank(checker):
    p = find(checker.check(["simvastatin", "clarithromycin", "metoprolol", "amiodarone"]),
             "metoprolol", "amiodarone")
    assert p.severity_basis == "derived"
    assert p.sources == ("drugbank",)
    assert p.mechanisms and p.severity == p.mechanisms[0].derived_severity


def test_known_limitation_opioid_benzo_not_major(checker):
    """Opioid + benzodiazepine carries an FDA boxed warning, but the KB only has the
    generic DrugBank template -> derived Moderate. Pinned so a fix is noticed
    (docs/limitations.md)."""
    p = find(checker.check(["oxycodone", "lorazepam"]), "lorazepam", "oxycodone")
    assert p.severity == "Moderate" and p.severity_basis == "derived"


def test_order_independent(checker):
    meds = ["coumadin", "aspirin", "lasix", "heparin"]
    r1, r2 = checker.check(meds), checker.check(meds[::-1])
    summary = lambda r: sorted((p.key, p.status, p.severity) for p in r.pairs)  # noqa: E731
    assert summary(r1) == summary(r2)


def test_duplicate_therapy_and_no_self_pair(checker):
    r = checker.check(["Percocet", "Tylenol 500 mg"])
    assert [d.concept.name for d in r.duplicates] == ["acetaminophen"]
    assert all(p.a.concept_id != p.b.concept_id for p in r.pairs)


def test_combination_components_not_paired_with_each_other(checker):
    r = checker.check(["Percocet"])
    assert r.pairs == []


def test_inactive_unmapped_class_and_non_drug_handling(checker):
    r = checker.check([
        Medication("warfarin"),
        Medication("simvastatin", status="held"),
        Medication("senna"),
        Medication("antibiotics"),
        Medication("NS"),
    ])
    assert [m.name for m in r.inactive] == ["simvastatin"]
    assert {m.name for m, _ in r.unchecked} == {"senna", "antibiotics"}
    assert r.pairs == []  # only warfarin is checkable


def test_no_known_requires_common_source(checker):
    r = checker.check(["acetaminophen", "furosemide"])
    p = find(r, "acetaminophen", "furosemide")
    assert p.status is PairStatus.NO_KNOWN_INTERACTION
    assert p.checked_sources == ("ddinter", "drugbank")


def test_not_covered_when_no_source_has_both(checker):
    kb = checker.kb
    dd_only = next(c for c in kb.concepts.values() if c.in_ddinter and not c.in_drugbank)
    db_only = next(c for c in kb.concepts.values()
                   if c.in_drugbank and not c.in_ddinter and not kb.lookup(c.concept_id, dd_only.concept_id))
    r = checker.check([dd_only.name, db_only.name])
    p = find(r, dd_only.name, db_only.name)
    assert p.status is PairStatus.NOT_COVERED and p.checked_sources == ()


def test_speed_on_realistic_list(checker):
    meds = ["warfarin", "aspirin", "furosemide", "metoprolol", "amiodarone", "lisinopril",
            "atorvastatin", "pantoprazole", "insulin glargine", "heparin", "ondansetron",
            "oxycodone", "lorazepam", "vancomycin", "piperacillin-tazobactam"]
    t0 = time.perf_counter()
    r = checker.check(meds)
    assert time.perf_counter() - t0 < 0.5
    # 16 concepts (pip-tazo = 2) -> 120 pairs, minus piperacillin+tazobactam (same product)
    assert len(r.pairs) == 16 * 15 // 2 - 1
