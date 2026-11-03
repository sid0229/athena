"""ML interaction predictor (Shtar et al., 2019 style): XGBoost on graph-similarity features.

Task: binary link prediction on the DrugBank DDI benchmark — given a drug pair,
predict whether an interaction is recorded (1) or not (0).

Features for a pair (u, v) are computed ONLY from the training interaction graph,
and for a training positive the edge (u, v) itself is removed before computing
them, so no label information leaks into its own features.

Two models (scripts/exp_ddi_upgrades.py chose them):
  warm  graph features + chemical structure (RDKit Morgan fingerprints and chemical-
        neighbour features) — for drugs that already have recorded interactions
  cold  chemical features only, trained with one or both drugs' own interactions
        hidden — for a drug with no recorded interactions (graph features fail there)

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

NBITS, K_NEIGHBOURS = 256, 5

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


# ---------------------------------------------------------------- chemical structure

def load_fingerprints(nbits: int = NBITS) -> dict:
    """drug id -> (RDKit fingerprint, numpy bit vector), Morgan radius 2."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    smi = pd.read_csv(data_path("drugbank_benchmark") / "drug_smiles.csv")
    out = {}
    for d, s in zip(smi.drug_id, smi.smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            fp = gen.GetFingerprint(m)
            out[d] = (fp, np.array(list(fp), dtype=np.uint8))
    return out


class ChemFeatures:
    """Pair features from chemical structure plus chemical similarity to known interactions."""

    def __init__(self, fingerprints: dict, g: dict[str, set[str]]):
        from rdkit import DataStructs

        self._bulk = DataStructs.BulkTanimotoSimilarity
        self.fp, self.g = fingerprints, g
        self.ids = list(fingerprints)
        self.known = [d for d in self.ids if g.get(d)]
        self._sim: dict[str, dict[str, float]] = {}
        self._knn: dict[str, list[str]] = {}

    def sims(self, d: str) -> dict[str, float]:
        if d not in self._sim:
            self._sim[d] = dict(zip(self.ids, self._bulk(self.fp[d][0], [self.fp[x][0] for x in self.ids])))
        return self._sim[d]

    def knn(self, d: str) -> list[str]:
        """Most similar drugs that have recorded interactions (excluding d itself)."""
        if d not in self._knn:
            s = self.sims(d)
            self._knn[d] = sorted((x for x in self.known if x != d), key=lambda x: -s[x])[:2 * K_NEIGHBOURS]
        return self._knn[d]

    def pair(self, u: str, v: str, hide: tuple[str, ...] = (), cold: bool = False) -> np.ndarray:
        """`hide`: drugs treated as having no recorded interactions (cold-start training/inference).
        warm layout: bits sum, bits product, tanimoto, max sim u->N(v), max sim v->N(u), max, kNN links
        cold layout: bits sum, bits product, tanimoto, max/min of the two neighbour sims, kNN links"""
        n = 2 * NBITS + (4 if cold else 5)
        if u not in self.fp or v not in self.fp:
            return np.zeros(n, dtype=np.float32)
        nb = lambda d, other: set() if d in hide else self.g.get(d, set()) - {other}  # noqa: E731
        su, sv = self.sims(u), self.sims(v)
        a = max((su[w] for w in nb(v, u) if w in su), default=0.0)
        b = max((sv[w] for w in nb(u, v) if w in sv), default=0.0)
        ku = [x for x in self.knn(u) if x not in hide][:K_NEIGHBOURS]
        kv = [x for x in self.knn(v) if x not in hide][:K_NEIGHBOURS]
        links = sum(1 for x in ku for y in kv if x != y and {x, y} != {u, v} and y in self.g.get(x, ()))
        bu, bv = self.fp[u][1], self.fp[v][1]
        tail = [su[v], max(a, b), min(a, b)] if cold else [su[v], a, b, max(a, b)]
        return np.concatenate([bu + bv, bu * bv, tail, [links / K_NEIGHBOURS ** 2]]).astype(np.float32)

    def featurize(self, pairs: pd.DataFrame, cold: bool = False, hides=None) -> np.ndarray:
        hides = hides or [()] * len(pairs)
        return np.stack([self.pair(u, v, h, cold) for u, v, h in zip(pairs["u"], pairs["v"], hides)])


CHEM_FEATURES = ([f"fp_sum_{i}" for i in range(NBITS)] + [f"fp_and_{i}" for i in range(NBITS)]
                 + ["tanimoto", "sim_u_to_partners_of_v", "sim_v_to_partners_of_u", "sim_to_partners_max",
                    "knn_links"])
