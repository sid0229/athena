"""Unit tests for Level 1 learning: retrieving similar reviewer corrections into LLM prompt."""

import json
from pathlib import Path

import pytest

from athena.extraction.corrections_fewshot import (
    CorrectionRetriever,
    build_db_corrections_bank,
    build_train_corrections_bank,
    med_dict_to_row,
)
from athena.extraction.llm_extractor import OllamaExtractor
from athena.extraction.pipeline import MedicationExtractor
from athena.review.corrections import CorrectionsDB


def test_med_dict_to_row():
    med = {
        "name": "Metoprolol Tartrate",
        "strength": "25 mg",
        "dosage": "1 tab",
        "form": "Tablet",
        "route": "PO",
        "frequency": "BID",
        "duration": "14 days",
        "status": "active",
    }
    row = med_dict_to_row(med)
    assert len(row) == 8
    assert row[0] == "Metoprolol Tartrate"
    assert row[1] == "25 mg"
    assert row[4] == "PO"
    assert row[5] == "BID"


def test_correction_retriever_tfidf_and_rapidfuzz():
    sample_bank = [
        ("Discharge Meds: metoprolol tartrate 25 mg PO BID for hypertension", [["metoprolol tartrate", "25 mg", "", "Tablet", "PO", "BID", "", ""]]),
        ("Patient on lisinopril 10 mg daily PO. Hold if SBP < 100.", [["lisinopril", "10 mg", "", "", "PO", "daily", "", ""]]),
        ("Warfarin 5 mg PO qPM target INR 2-3.", [["warfarin", "5 mg", "", "", "PO", "qPM", "", ""]]),
    ]
    retriever_tfidf = CorrectionRetriever(bank=sample_bank, k=2, method="tfidf")
    assert retriever_tfidf.version.startswith("v_")
    assert len(retriever_tfidf.bank) == 3

    # Metoprolol query matches metoprolol snippet
    results = retriever_tfidf.retrieve("metoprolol 50 mg PO BID", k=1)
    assert len(results) == 1
    assert "metoprolol" in results[0][0].lower()

    # Rapidfuzz method
    retriever_rf = CorrectionRetriever(bank=sample_bank, k=2, method="rapidfuzz")
    results_rf = retriever_rf.retrieve("lisinopril 20 mg PO daily", k=1)
    assert len(results_rf) == 1
    assert "lisinopril" in results_rf[0][0].lower()


def test_empty_retriever_handles_gracefully():
    retriever = CorrectionRetriever(bank=[], k=2)
    assert retriever.version == "v0_empty"
    assert retriever.retrieve("any text") == []


def test_db_corrections_bank(tmp_path: Path):
    db_path = tmp_path / "test_corr.db"
    db = CorrectionsDB(db_path)
    entry = db.record(
        note_hash="hash123",
        original_extraction=[],
        corrected_list=[{"name": "Aspirin", "strength": "81 mg", "route": "PO", "frequency": "daily"}],
        reviewer="Dr. Smith",
        case_id="case-1",
        note_text="Discharge Medications:\n1. Aspirin 81 mg PO daily\n",
    )
    assert entry.note_text == "Discharge Medications:\n1. Aspirin 81 mg PO daily\n"

    bank = build_db_corrections_bank(db_path)
    assert len(bank) == 1
    assert "Aspirin" in bank[0][0]
    assert bank[0][1][0][0] == "Aspirin"
    assert bank[0][1][0][1] == "81 mg"


def test_cache_signature_includes_bank_version():
    sample_bank = [
        ("metoprolol 25 mg PO BID", [["metoprolol", "25 mg", "", "", "PO", "BID", "", ""]]),
    ]
    retriever1 = CorrectionRetriever(bank=sample_bank)
    llm1 = OllamaExtractor(use_cache=False, use_corrections=True, retriever=retriever1)
    sig1 = llm1._cache_signature()

    assert sig1["use_corrections"] is True
    assert "correction_bank_version" in sig1
    assert sig1["correction_bank_version"] == retriever1.version
    assert f"_corr_{retriever1.version}" in str(llm1.cache_dir)

    # When disabled, signature matches baseline and has no bank version
    llm_baseline = OllamaExtractor(use_cache=False, use_corrections=False)
    sig_base = llm_baseline._cache_signature()
    assert "use_corrections" not in sig_base
    assert "correction_bank_version" not in sig_base
    assert "_corr_" not in str(llm_baseline.cache_dir)

    # Different bank -> different version and cache path
    sample_bank2 = [
        ("atorvastatin 40 mg PO nightly", [["atorvastatin", "40 mg", "", "", "PO", "nightly", "", ""]]),
    ]
    retriever2 = CorrectionRetriever(bank=sample_bank2)
    llm2 = OllamaExtractor(use_cache=False, use_corrections=True, retriever=retriever2)
    sig2 = llm2._cache_signature()
    assert sig2["correction_bank_version"] != sig1["correction_bank_version"]
    assert llm1._cache_path("some chunk") != llm2._cache_path("some chunk")


def test_leakage_free_train_bank():
    # Only loads notes from train split
    bank = build_train_corrections_bank(split="train", limit=3)
    assert len(bank) > 0
    for snippet, rows in bank:
        assert isinstance(snippet, str) and len(snippet) > 0
        assert isinstance(rows, list) and len(rows) > 0
        assert len(rows[0]) == 8


def test_medication_extractor_wiring():
    sample_bank = [
        ("metoprolol 25 mg PO BID", [["metoprolol", "25 mg", "", "", "PO", "BID", "", ""]]),
    ]
    retriever = CorrectionRetriever(bank=sample_bank)
    ex = MedicationExtractor("rules", use_corrections=True, retriever=retriever)
    assert ex.use_corrections is True
