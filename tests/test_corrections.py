"""Tests for medication list corrections, server endpoint, and audit chain verification."""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from athena.config import ROOT
from athena.fusion.report import analyze, analyze_medications
from athena.review.audit import AuditLog
from athena.review.corrections import CorrectionsDB
from athena.verification.checker import InteractionChecker


@pytest.fixture
def tmp_dbs(tmp_path):
    audit_path = tmp_path / "audit.db"
    corr_path = tmp_path / "corrections.db"
    return AuditLog(audit_path), CorrectionsDB(corr_path)


def test_correction_changes_report():
    checker = InteractionChecker()
    meds = [
        {"name": "warfarin", "status": "active", "attributes": {"Strength": "5 mg"}},
        {"name": "aspirin", "status": "active", "attributes": {"Strength": "81 mg"}},
    ]
    rep1 = analyze_medications(meds, checker=checker)
    assert any(f.type == "INTERACTION" for f in rep1.findings)
    assert len(rep1.medications) == 2

    # Remove aspirin: only warfarin remains -> no interaction
    meds_corrected = [
        {"name": "warfarin", "status": "active", "attributes": {"Strength": "5 mg"}}
    ]
    rep2 = analyze_medications(meds_corrected, checker=checker)
    assert not any(f.type == "INTERACTION" for f in rep2.findings)
    assert len(rep2.medications) == 1
    assert rep1.report_hash != rep2.report_hash


def test_added_drug_gets_interaction_checked():
    checker = InteractionChecker()
    # Initially only warfarin
    meds_initial = [
        {"name": "warfarin", "status": "active", "attributes": {"Strength": "5 mg"}}
    ]
    rep_initial = analyze_medications(meds_initial, checker=checker)
    assert not any(f.type == "INTERACTION" for f in rep_initial.findings)

    # Reviewer adds missed drug: clarithromycin
    meds_with_missed = meds_initial + [
        {"name": "clarithromycin", "status": "active", "attributes": {"Strength": "500 mg", "Route": "PO", "Frequency": "BID"}}
    ]
    rep_corrected = analyze_medications(meds_with_missed, checker=checker)

    # Finding for warfarin + clarithromycin should be present and Critical
    int_findings = [f for f in rep_corrected.findings if f.type == "INTERACTION"]
    assert len(int_findings) >= 1
    waf_clari = next(f for f in int_findings if "warfarin" in f.title.lower() and "clarithromycin" in f.title.lower())
    assert waf_clari.tier == "critical"
    assert waf_clari.risk >= 0.7


def test_audit_chain_verifies_after_correction(tmp_path):
    audit_db = AuditLog(tmp_path / "audit.db")
    corr_db = CorrectionsDB(tmp_path / "corrections.db")

    # Record regular confirm
    audit_db.record(
        reviewer="Dr Rao", case_id="case-1", report_hash="h1" * 32,
        finding_id="INT:1|2", finding_type="INTERACTION", tier="critical",
        title="Major interaction: a + b", action="confirm"
    )

    # Record correction
    corr_db.record(
        note_hash="nh" * 32,
        original_extraction=[{"name": "warfarin"}],
        corrected_list=[{"name": "warfarin"}, {"name": "aspirin"}],
        reviewer="Dr Rao",
        case_id="case-1"
    )
    audit_db.record(
        reviewer="Dr Rao", case_id="case-1", report_hash="h2" * 32,
        finding_id="CORRECTION", finding_type="MEDICATION_CORRECTION", tier="info",
        title="Medication list corrected (2 medications)", action="correct",
        comment="Added missed aspirin"
    )

    # Verify audit chain
    ok, bad = audit_db.verify()
    assert ok is True
    assert bad is None
    entries = audit_db.entries()
    assert len(entries) == 2
    assert entries[1].action == "correct"
    assert entries[1].prev_hash == entries[0].row_hash


def test_server_correct_endpoint(tmp_path):
    os.environ["ATHENA_AUDIT_DB"] = str(tmp_path / "audit.db")
    os.environ["ATHENA_CORRECTIONS_DB"] = str(tmp_path / "corrections.db")
    sys.path.insert(0, str(ROOT / "app"))
    import importlib
    server = importlib.import_module("server")
    importlib.reload(server)
    client = TestClient(server.app)

    # Analyze note first
    note = "Discharge Medications:\n1. warfarin 5 mg PO daily\n"
    res1 = client.post("/api/analyze", json={"text": note, "mode": "rules"})
    assert res1.status_code == 200
    a1 = res1.json()
    h1 = a1["report"]["meta"]["report_hash"]

    # No interactions initially
    assert not any(f["type"] == "INTERACTION" for f in a1["report"]["findings"])

    # Correct list by adding aspirin
    corrected_meds = [
        {"name": "warfarin", "status": "active", "attributes": {"Strength": "5 mg", "Route": "PO"}},
        {"name": "aspirin", "status": "active", "attributes": {"Strength": "81 mg", "Route": "PO"}},
    ]
    res2 = client.post("/api/correct", json={
        "report_hash": h1,
        "reviewer": "Dr Reviewer",
        "medications": corrected_meds,
        "comment": "Added aspirin 81 mg"
    })
    assert res2.status_code == 200
    a2 = res2.json()
    h2 = a2["report"]["meta"]["report_hash"]
    assert h1 != h2

    # Check interaction is now present
    ints = [f for f in a2["report"]["findings"] if f["type"] == "INTERACTION"]
    assert len(ints) >= 1
    assert any("warfarin" in f["title"].lower() and "acetylsalicylic acid" in f["title"].lower() for f in ints)

    # Check audit log verification through server API
    audit_res = client.get("/api/audit")
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["chain_ok"] is True
    assert any(e["action"] == "correct" for e in audit_data["entries"])

    # Check corrections DB stored the correction
    corr_entries = server.corrections().entries()
    assert len(corr_entries) == 1
    assert corr_entries[0].reviewer == "Dr Reviewer"
    assert len(corr_entries[0].corrected_list) == 2
