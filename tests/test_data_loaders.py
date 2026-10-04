"""Batch 1: data loader tests. Counts were verified against the raw files on 4 Oct 2026."""

import pytest

from athena.data import ddinter, drugbank, mimic, n2c2


# ---------- n2c2 ----------

def test_parse_ann_handles_discontiguous_and_newline_text():
    ann = (
        "T1\tDrug 10 20;25 30\tabc def\n"
        "T2\tFrequency 40 52\tevery\n"
        "12 hours\n"
        "R1\tFrequency-Drug Arg1:T2 Arg2:T1\n"
        "#1\tAnnotatorNotes T1\tnote\n"
    )
    ents, rels = n2c2.parse_ann(ann)
    assert ents["T1"].spans == ((10, 20), (25, 30))
    assert ents["T2"].text == "every\n12 hours"
    assert rels == [n2c2.Relation("R1", "Frequency-Drug", "T2", "T1")]


@pytest.mark.parametrize(
    "split, n_notes, n_drugs, n_entities, n_relations",
    [
        ("train", 265, 14311, 44248, 31325),
        ("val", 38, 2046, 6192, 4281),
        ("test", 202, 10575, 32918, 23462),
    ],
)
def test_n2c2_split_counts(split, n_notes, n_drugs, n_entities, n_relations):
    notes = n2c2.load_split(split)
    assert len(notes) == n_notes
    assert sum(len(n.drugs()) for n in notes) == n_drugs
    assert sum(len(n.entities) for n in notes) == n_entities
    assert sum(len(n.relations) for n in notes) == n_relations


def test_n2c2_single_span_offsets_match_text():
    """Offsets are authoritative: every contiguous entity's text equals the note slice."""
    for split in n2c2.SPLIT_ARCHIVES:
        for note in n2c2.load_split(split):
            for e in note.entities.values():
                if len(e.spans) == 1:
                    a, b = e.spans[0]
                    assert note.text[a:b].split() == e.text.split(), (note.id, e.id)


def test_n2c2_relations_resolve_and_point_to_drugs():
    """Every relation targets a Drug. The source annotations contain 18 relations
    (all in train) whose label disagrees with the attribute's entity type, e.g.
    'ADE-Drug' pointing at a Reason entity; that known noise is pinned here."""
    mislabelled = 0
    for split in n2c2.SPLIT_ARCHIVES:
        for note in n2c2.load_split(split):
            for r in note.relations:
                assert note.entities[r.arg2].type == "Drug"
                mislabelled += note.entities[r.arg1].type != r.type.split("-")[0]
    assert mislabelled == 18


def test_n2c2_splits_are_disjoint():
    ids = {s: {n.id for n in n2c2.load_split(s)} for s in n2c2.SPLIT_ARCHIVES}
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert not ids["train"] & ids["val"]


# ---------- MIMIC-III demo ----------

def test_mimic_prescriptions():
    rx = mimic.load_prescriptions()
    assert len(rx) == 10398
    assert rx["hadm_id"].nunique() == 122
    assert rx["drug_name"].notna().all()


def test_mimic_admission_med_lists():
    lists = mimic.admission_med_lists()
    assert len(lists) == 122
    assert all(meds == sorted(set(meds)) for meds in lists.values())


# ---------- DrugBank ----------

def test_kaggle_ddi():
    df = drugbank.load_kaggle_ddi()
    assert len(df) == 191541
    assert df["template"].nunique() == 86
    assert df["template"].str.contains("{A}", regex=False).all()
    assert df["template"].str.contains("{B}", regex=False).all()


def test_template_prefers_longer_name():
    t = drugbank.description_to_template("Iron sucrose may decrease Iron.", "Iron", "Iron sucrose")
    assert t == "{B} may decrease {A}."


def test_benchmark_ddi():
    df = drugbank.load_benchmark_ddi()
    assert len(df) == 191808
    assert df["type"].nunique() == 86
    assert len(drugbank.load_smiles()) == 1706


# ---------- DDInter ----------

def test_ddinter_dedup_and_levels():
    df = ddinter.load_ddinter()
    assert len(df) == 160235
    assert (df["id_a"] < df["id_b"]).all()
    assert not df.duplicated(["id_a", "id_b"]).any()
    assert set(df["level"].cat.categories) == set(ddinter.SEVERITY_ORDER)
    assert len(ddinter.ddinter_drugs()) == 1939


def _ddinter_pair(a: str, b: str):
    df = ddinter.load_ddinter()
    m = ((df.drug_a == a) & (df.drug_b == b)) | ((df.drug_a == b) & (df.drug_b == a))
    return df[m]


def test_ddinter_known_pairs():
    assert _ddinter_pair("Heparin", "Ketorolac")["level"].item() == "Moderate"
    assert _ddinter_pair("Abciximab", "Heparin")["level"].item() == "Major"
