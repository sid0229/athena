"""Batch 2a: drug lexicon + normaliser. Regression cases come from the manual audit."""

import collections

import pytest

from athena.normalize.lexicon import load_lexicon
from athena.normalize.normalizer import DrugNormalizer


@pytest.fixture(scope="module")
def norm():
    try:
        load_lexicon()
    except FileNotFoundError:
        pytest.skip("run scripts/build_lexicon.py first")
    return DrugNormalizer()


@pytest.mark.parametrize(
    "raw, expected",
    [
        # brands / abbreviations
        ("ASA", ("acetylsalicylic acid",)),
        ("aspirin ec", ("acetylsalicylic acid",)),
        ("Coumadin", ("warfarin",)),
        ("Lasix", ("furosemide",)),
        ("Dilaudid", ("hydromorphone",)),
        ("vanc", ("vancomycin",)),
        ("ntg sl", ("nitroglycerin",)),
        ("albuterol 0.083% neb soln", ("salbutamol",)),
        # salts / formulations
        ("metoprolol tartrate", ("metoprolol",)),
        ("metoprolol succinate xl", ("metoprolol",)),
        ("vancomycin hcl", ("vancomycin",)),
        ("heparin sodium", ("heparin",)),
        ("hydrocortisone na succ.", ("hydrocortisone",)),
        ("morphine sulfate (syringe)", ("morphine",)),
        ("megestrol oral suspension", ("megestrol acetate",)),
        # salt word at the start is part of the name
        ("potassium chloride", ("potassium chloride",)),
        # combinations
        ("Percocet", ("acetaminophen", "oxycodone")),
        ("piperacillin-tazobactam", ("piperacillin", "tazobactam")),
        ("carbidopa-levodopa (25-100)", ("carbidopa", "levodopa")),
    ],
)
def test_maps_to_expected(norm, raw, expected):
    r = norm.normalize(raw)
    assert r.status == "mapped", r
    assert r.drugs == expected


@pytest.mark.parametrize(
    "raw",
    [
        "sodium polystyrene sulfonate",  # was wrongly -> tolevamer
        "albumin, human",                # was wrongly -> perflutren
        "albumin 25% (12.5g / 50ml)",
    ],
)
def test_audited_false_matches_stay_fixed(norm, raw):
    r = norm.normalize(raw)
    assert "tolevamer" not in r.drugs and "perflutren" not in r.drugs


def test_megestrol_is_not_nomegestrol(norm):
    assert norm.normalize("megestrol").drugs == ("megestrol acetate",)


@pytest.mark.parametrize("raw", ["d5w", "NS", "sw", "sodium chloride 0.9%  flush", "vial", "FFP", "prbc", "IVF"])
def test_non_drugs(norm, raw):
    assert norm.normalize(raw).status == "non_drug"


@pytest.mark.parametrize("raw", ["antibiotics", "steroids", "PPI", "beta blocker", "statin", "narcotics"])
def test_drug_classes(norm, raw):
    assert norm.normalize(raw).status == "drug_class"


def test_unknown_string_is_unmapped_not_guessed(norm):
    r = norm.normalize("zzqx-unknown-compound")
    assert r.status == "unmapped" and r.drugs == () and r.score == 0.0


def test_lexicon_name_to_id_validated():
    lex = load_lexicon()
    assert len(lex["kaggle_name_to_id"]) >= 1670  # 1678 when built (98.6% of 1,701)
    assert lex["kaggle_name_to_id"]["Warfarin"] == "DB00682"
    assert lex["kaggle_name_to_id"]["Carbamazepine"] == "DB00564"


def test_lexicon_ids_agree_with_benchmark_graph():
    """Mapped Kaggle pairs must exist in the ID-based benchmark (same underlying data)."""
    from athena.data import drugbank

    n2id = load_lexicon()["kaggle_name_to_id"]
    kag = drugbank.load_kaggle_ddi()
    bench = drugbank.load_benchmark_ddi()
    bench_pairs = {frozenset(p) for p in zip(bench["id_a"], bench["id_b"])}
    pairs = [(n2id.get(a), n2id.get(b)) for a, b in zip(kag["drug_a"], kag["drug_b"])]
    mapped = [p for p in pairs if all(p)]
    agree = sum(frozenset(p) in bench_pairs for p in mapped) / len(mapped)
    assert agree > 0.999


def test_mimic_coverage_floor(norm):
    from athena.data import mimic

    counts = collections.Counter(mimic.load_prescriptions()["drug_name"].str.lower())
    status = collections.Counter()
    for name, n in counts.items():
        status[norm.normalize(name).status] += n
    specific = status["mapped"] + status["unmapped"] + status["ambiguous"]
    assert status["mapped"] / specific >= 0.88  # 90.9% when built
