"""Batch 4: chunking, grounding, slot repair, scoring (no LLM calls)."""

from athena.data.n2c2 import Entity, Note, Relation
from athena.eval.n2c2_metrics import score_note
from athena.extraction.align import Mention, Span, ground
from athena.extraction.chunking import chunk_text
from athena.extraction.llm_extractor import RawMed
from athena.extraction.slot_repair import repair

TEXT = (
    "Discharge Medications:\n"
    "1. metoprolol tartrate 25 mg Tablet Sig: One (1) Tablet PO BID (2 times a day).\n"
    "2. Lasix 40 mg Tablet Sig: One (1) Tablet PO DAILY (Daily).\n"
    "Lasix was held on admission. Continue metoprolol.\n"
)


def test_chunking_covers_text_with_overlap():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(200))
    chunks = chunk_text(text, 1000, 200)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap
        assert text[b.start - 1] == "\n"  # starts on a line
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_ground_attributes_and_propagation():
    chunk = chunk_text(TEXT, 4000, 300)[0]
    rows = [
        RawMed("metoprolol tartrate", "25 mg", "One (1)", "Tablet", "PO", "BID"),
        RawMed("Lasix", "40 mg", "One (1)", "Tablet", "PO", "DAILY", status="held"),
        RawMed("metoprolol", status="continued"),
    ]
    mentions, stats = ground(TEXT, [(chunk, rows)])
    by_text = {}
    for m in mentions:
        by_text.setdefault(m.drug.text, []).append(m)
    assert len(by_text["Lasix"]) == 2  # both occurrences found
    first = by_text["metoprolol tartrate"][0]
    assert [s.text for s in first.attributes["Strength"]] == ["25 mg"]
    assert {s.text for s in first.attributes["Form"]} == {"Tablet"}
    assert len(first.attributes["Form"]) == 2  # n2c2 labels both "Tablet"s
    assert first.attributes["Frequency"][0].text == "BID (2 times a day)"  # n2c2 extension
    assert by_text["Lasix"][0].attributes["Frequency"][0].text == "DAILY (Daily)"
    assert stats.values_dropped == []


def test_hallucinated_values_and_drugs_are_dropped():
    chunk = chunk_text(TEXT, 4000, 300)[0]
    rows = [RawMed("Lasix", "80 mg", route="IV"), RawMed("warfarin", "5 mg")]
    mentions, stats = ground(TEXT, [(chunk, rows)])
    assert all(m.drug.text.lower() != "warfarin" for m in mentions)
    assert stats.drugs_not_found == 1
    assert {(f, v) for _, f, v in stats.values_dropped} == {("strength", "80 mg"), ("route", "IV")}


def test_attributes_stay_within_own_item():
    chunk = chunk_text(TEXT, 4000, 300)[0]
    # LLM wrongly gives metoprolol Lasix's strength: 40 mg is outside metoprolol's window.
    mentions, stats = ground(TEXT, [(chunk, [RawMed("metoprolol tartrate", "40 mg"), RawMed("Lasix")])])
    meto = next(m for m in mentions if m.drug.text == "metoprolol tartrate")
    assert "Strength" not in meto.attributes
    assert ("metoprolol tartrate", "strength", "40 mg") in stats.values_dropped


def test_slot_repair_fixes_column_shift_only():
    fixed, moved = repair(RawMed("Effexor", "One (1)", "Tablet", "PO", "daily"))
    assert (fixed.dosage, fixed.form, fixed.route, fixed.frequency) == ("One (1)", "Tablet", "PO", "daily")
    assert moved == 4
    ok = RawMed("metoprolol", "25 mg", "One (1)", "Tablet", "PO", "BID (2 times a day)")
    assert repair(ok) == (ok, 0)


def test_scoring_strict_vs_lenient():
    note = Note("n1", "test", "Lasix 40 mg PO daily", entities={
        "T1": Entity("T1", "Drug", ((0, 5),), "Lasix"),
        "T2": Entity("T2", "Strength", ((6, 11),), "40 mg"),
        "T3": Entity("T3", "Frequency", ((15, 20),), "daily"),
    }, relations=[Relation("R1", "Strength-Drug", "T2", "T1")])
    pred = [Mention(Span(0, 5, "Lasix"), {"Strength": [Span(6, 11, "40 mg")],
                                         "Frequency": [Span(12, 20, "PO daily")]})]
    s = score_note(note, pred)
    assert s.strict["Drug"].tp == 1 and s.strict["Strength"].tp == 1
    assert s.strict["Frequency"].tp == 0 and s.lenient["Frequency"].tp == 1
    assert s.rel_strict.tp == 1


class _OfflineLLM:
    """Stands in for Ollama being down."""
    def available(self):
        return False


def test_rules_mode_finds_drugs_and_attributes():
    from athena.extraction.pipeline import MedicationExtractor

    r = MedicationExtractor("rules").extract(TEXT)
    drugs = [m.drug.text for m in r.mentions]
    assert "metoprolol tartrate" in drugs and drugs.count("Lasix") == 2
    meto = next(m for m in r.mentions if m.drug.text == "metoprolol tartrate")
    assert [s.text for s in meto.attributes["Strength"]] == ["25 mg"]
    assert meto.attributes["Route"][0].text == "PO"
    assert meto.attributes["Frequency"][0].text == "BID (2 times a day)"
    assert meto.attributes["Dosage"][0].text == "One (1)"


def test_hybrid_falls_back_to_rules_when_llm_unavailable():
    from athena.extraction.pipeline import MedicationExtractor

    r = MedicationExtractor("hybrid", llm=_OfflineLLM()).extract(TEXT)
    assert r.mode == "rules" and r.warnings
    assert any(m.drug.text == "Lasix" for m in r.mentions)


def test_lab_sections_are_skipped_by_dictionary():
    from athena.extraction.pipeline import MedicationExtractor

    text = "Pertinent Results:\nPotassium 4.1, glucose 110, heparin level 0.4\n\nDischarge Medications:\n1. heparin 5000 units SC TID\n"
    r = MedicationExtractor("rules").extract(text)
    assert [m.drug.text for m in r.mentions] == ["heparin"]
    assert r.mentions[0].drug.start > text.index("Discharge Medications")
