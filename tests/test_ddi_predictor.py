"""ML interaction predictor: feature leakage, negative sampling, saved results."""

import json

import pandas as pd

from athena.config import ROOT
from athena.verification.ddi_predictor import FEATURES, build_graph, pair_features, with_negatives


def test_target_edge_is_excluded_from_its_own_features():
    g = build_graph(pd.DataFrame({"d1": ["A", "A", "B"], "d2": ["B", "C", "C"]}))
    f = dict(zip(FEATURES, pair_features(g, "A", "B")))
    # A-B is a training edge: degrees exclude it, C is the only common neighbour
    assert f["deg_u"] == 1 and f["deg_v"] == 1 and f["common_neighbors"] == 1


def test_negatives_follow_benchmark_and_skip_true_interactions():
    df = pd.DataFrame({"d1": ["A", "A"], "d2": ["B", "C"], "Neg samples": ["D$t", "B$t"]})
    out = with_negatives(df, positives={frozenset("AB"), frozenset("AC")})
    neg = out[out.label == 0]
    assert list(zip(neg.u, neg.v)) == [("A", "D")]  # A-B dropped: it is a real interaction


def test_saved_results_meet_floor():
    res = json.loads((ROOT / "docs" / "results" / "ddi_xgboost.json").read_text())
    x, b = res["xgboost"], res["baseline_logistic_regression"]
    assert x["f1"] > 0.8 and x["roc_auc"] > 0.9
    assert x["roc_auc"] > b["roc_auc"]  # beats the linear baseline
