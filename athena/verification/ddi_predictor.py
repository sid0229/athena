"""ML interaction predictor (Shtar et al., 2019 style): XGBoost on graph-similarity features.

Task: binary link prediction on the DrugBank DDI benchmark — given a drug pair,
predict whether an interaction is recorded (1) or not (0).

Features for a pair (u, v) are computed ONLY from the training interaction graph,
and for a training positive the edge (u, v) itself is removed before computing
them, so no label information leaks into its own features.

This is a research component (Phase 2 preview). Its predictions are never shown
in the dashboard as verified interactions; Athena's flags still come only from
the curated knowledge base (README, "Design principles").
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from athena.config import data_path

FEATURES = [
    "deg_u", "deg_v", "deg_min", "deg_max",
    "common_neighbors", "jaccard", "adamic_adar", "resource_allocation",
    "pref_attachment", "sorensen", "hub_promoted", "hub_depressed",
]


def load_split(setting: str = "warm_start", fold: str = "fold0") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = data_path("drugbank_benchmark") / setting / fold
    return pd.read_csv(base / "train.csv"), pd.read_csv(base / "test.csv")


def with_negatives(df: pd.DataFrame, positives: set[frozenset]) -> pd.DataFrame:
    """Positive rows plus one benchmark-provided negative per positive.

    'DBxxxx$t' replaces the second drug, 'DBxxxx$h' the first. Negatives that are
    in fact recorded interactions anywhere in the benchmark are dropped.
    """
    pos = pd.DataFrame({"u": df["d1"], "v": df["d2"], "label": 1})
    neg_rows = []
    for d1, d2, neg in zip(df["d1"], df["d2"], df["Neg samples"]):
        drug, side = str(neg).split("$")
        u, v = (d1, drug) if side == "t" else (drug, d2)
        if u != v and frozenset((u, v)) not in positives:
            neg_rows.append((u, v, 0))
    neg = pd.DataFrame(neg_rows, columns=["u", "v", "label"])
    return pd.concat([pos, neg], ignore_index=True)


def build_graph(train_pos: pd.DataFrame) -> dict[str, set[str]]:
    g: dict[str, set[str]] = defaultdict(set)
    for a, b in zip(train_pos["d1"], train_pos["d2"]):
        g[a].add(b)
        g[b].add(a)
    return g


def pair_features(g: dict[str, set[str]], u: str, v: str) -> list[float]:
    nu = g.get(u, set()) - {v}  # remove the target edge itself (no leakage)
    nv = g.get(v, set()) - {u}
    du, dv = len(nu), len(nv)
    common = nu & nv
    cn = len(common)
    union = len(nu | nv)
    aa = sum(1 / math.log(len(g[w])) for w in common if len(g[w]) > 1)
    ra = sum(1 / len(g[w]) for w in common if g[w])
    return [
        du, dv, min(du, dv), max(du, dv),
        cn, cn / union if union else 0.0, aa, ra,
        du * dv, 2 * cn / (du + dv) if du + dv else 0.0,
        cn / min(du, dv) if min(du, dv) else 0.0,
        cn / max(du, dv) if max(du, dv) else 0.0,
    ]


def featurize(g: dict[str, set[str]], pairs: pd.DataFrame) -> np.ndarray:
    return np.array([pair_features(g, u, v) for u, v in zip(pairs["u"], pairs["v"])], dtype=np.float32)
