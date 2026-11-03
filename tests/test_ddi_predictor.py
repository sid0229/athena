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


def test_chemical_features_exclude_target_edge():
    import numpy as np
    from athena.verification.ddi_predictor import ChemFeatures, load_fingerprints
    fps = load_fingerprints()
    a, b, c = list(fps)[:3]
    g = build_graph(pd.DataFrame({"d1": [a], "d2": [b]}))
    f = ChemFeatures(fps, g).pair(a, b)
    # the only partner of b is a itself: similarity-to-partners must not see the target edge
    assert f[-5 + 1] == 0 and f[-5 + 2] == 0
    cold = ChemFeatures(fps, g).pair(a, c, hide=(a,), cold=True)
    assert cold.shape == (2 * 256 + 4,) and np.isfinite(cold).all()


def test_saved_results_chemistry_helps_and_cold_start_reported():
    res = json.loads((ROOT / "docs" / "results" / "ddi_xgboost.json").read_text())
    assert res["xgboost"]["roc_auc"] > res["xgboost_graph_only"]["roc_auc"]
    s2 = res["cold_start"]["S2"]
    assert s2["xgboost_cold_chemistry"]["roc_auc"] > max(0.7, s2["xgboost_graph_only"]["roc_auc"])
