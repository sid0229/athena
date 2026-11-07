"""Experiment: does adding chemical structure help the XGBoost DDI predictor, and how does it
hold up under realistic class imbalance and on unseen (cold-start) drugs?

    python scripts/exp_ddi_upgrades.py

Feature sets (all computed from the TRAINING interaction graph only):
  graph        12 graph-similarity features (current model, Shtar et al. 2019 style)
  chem         RDKit Morgan fingerprints (r=2, 256 bits): sum and product of the pair's bits,
               Tanimoto(u, v), max Tanimoto of u to v's known partners (and vice versa),
               and interactions recorded between the 5 nearest chemical neighbours of u and of v
  graph+chem   both

Settings: warm-start fold0 (both drugs seen in training, new pairs) and cold-start fold0
(S2 = one drug unseen in training, S1 = both unseen). Evaluation at 1:1 (benchmark
negatives), at the benchmark's own density (~1:7) and at a 1:50 stress ratio.
Writes docs/results/ddi_upgrades.json.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from athena.config import ROOT, data_path  # noqa: E402
from athena.data.drugbank import load_benchmark_ddi  # noqa: E402
from athena.verification.ddi_predictor import (ChemFeatures, build_graph, featurize, load_fingerprints,  # noqa: E402
                                               with_negatives)

SEED = 42
BASE = data_path("drugbank_benchmark")


# ------------------------------------------------------------------ chemistry
def load_fps():
    """Shared implementation (athena.verification.ddi_predictor); returns (fps, bits) for this script."""
    fp = load_fingerprints()
    return {d: v[0] for d, v in fp.items()}, fp


class Chem2(ChemFeatures):
    """The experiment's feature calls on top of the shipped ChemFeatures (the earlier private copy had
    no kNN buffer in cold mode, so its numbers came from slightly different code — code review)."""

    def __init__(self, fps, bits, g):
        super().__init__(bits, g)

    def pair_cold(self, u, v, hide=()):
        return self.pair(u, v, hide, cold=True)


def cold_negs(df):
    """Cold-start files keep the negative sample in the 'split' column."""
    return df.assign(**{"Neg samples": df["split"]})


def imbalanced(pos_df, all_pos, drugs, ratio, n_pos, rng):
    """n_pos test positives + ratio corrupted negatives each (one side replaced at random)."""
    pos = pos_df.sample(min(n_pos, len(pos_df)), random_state=SEED)
    rows = [(u, v, 1) for u, v in zip(pos.d1, pos.d2)]
    for u, v in zip(pos.d1, pos.d2):
        made = 0
        while made < ratio:
            w = rng.choice(drugs)
            a, b = (u, w) if rng.random() < 0.5 else (w, v)
            if a != b and frozenset((a, b)) not in all_pos:
                rows.append((a, b, 0))
                made += 1
    return pd.DataFrame(rows, columns=["u", "v", "label"])


def scores(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "prevalence": round(float(y.mean()), 4)}


def fit(X, y):
    m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                          eval_metric="logloss", early_stopping_rounds=40, n_jobs=4, random_state=SEED,
                          tree_method="hist")
    idx = np.random.RandomState(SEED).permutation(len(y))
    cut = int(0.9 * len(y))
    m.fit(X[idx[:cut]], y[idx[:cut]], eval_set=[(X[idx[cut:]], y[idx[cut:]])], verbose=False)
    return m


def run_setting(name, train_df, tests, all_pos, fps, bits):
    t0 = time.time()
    g = build_graph(train_df)
    ch = Chem2(fps, bits, g)
    train = with_negatives(train_df, all_pos)
    y = train.label.to_numpy()
    Xg, Xc = featurize(g, train), ch.featurize(train)
    print(f"[{name}] train {len(train):,} pairs, features in {time.time() - t0:.0f}s")
    sets = {"graph": Xg, "chem": Xc, "graph+chem": np.hstack([Xg, Xc])}
    models = {k: fit(X, y) for k, X in sets.items()}
    print(f"[{name}] models trained ({time.time() - t0:.0f}s)")
    out = {}
    for tname, tdf in tests.items():
        tg, tc = featurize(g, tdf), ch.featurize(tdf)
        tsets = {"graph": tg, "chem": tc, "graph+chem": np.hstack([tg, tc])}
        out[tname] = {k: scores(tdf.label.to_numpy(), models[k].predict_proba(tsets[k])[:, 1]) for k in sets}
        print(f"[{name}] {tname:28s} " + " | ".join(
            f"{k}: AUC {v['roc_auc']:.3f} PR {v['pr_auc']:.3f} F1 {v['f1']:.3f}" for k, v in out[tname].items()))
    return out, models


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    bench = load_benchmark_ddi()
    all_pos = {frozenset(p) for p in zip(bench.id_a, bench.id_b)}
    n_drugs = len(set(bench.id_a) | set(bench.id_b))
    density = len(all_pos) / (n_drugs * (n_drugs - 1) / 2)
    real_ratio = round((1 - density) / density)
    fps, bits = load_fps()
    print(f"{len(fps)} drugs with valid SMILES; benchmark density {density:.3f} (~1:{real_ratio})")

    res = {"benchmark_density": round(density, 4), "realistic_ratio": f"1:{real_ratio}", "settings": {}}

    # warm start
    tr = pd.read_csv(BASE / "warm_start/fold0/train.csv")
    te = pd.read_csv(BASE / "warm_start/fold0/test.csv")
    trp = {frozenset(p) for p in zip(tr.d1, tr.d2)}
    te = te[[frozenset(p) not in trp for p in zip(te.d1, te.d2)]]
    seen = sorted(set(tr.d1) | set(tr.d2))
    tests = {"1:1 (benchmark negatives)": with_negatives(te, all_pos),
             f"1:{real_ratio} (benchmark density)": imbalanced(te, all_pos, seen, real_ratio, 4000, rng),
             "1:50 (stress test)": imbalanced(te, all_pos, seen, 50, 2000, rng)}
    res["settings"]["warm_start"], warm_models = run_setting("warm", tr, tests, all_pos, fps, bits)

    # cold start: S2 one unseen drug, S1 both unseen
    tr = cold_negs(pd.read_csv(BASE / "cold_start/fold0/train.csv"))
    s1 = cold_negs(pd.read_csv(BASE / "cold_start/fold0/s1.csv"))
    s2 = cold_negs(pd.read_csv(BASE / "cold_start/fold0/s2.csv"))
    drugs = sorted(set(bench.id_a) | set(bench.id_b))
    tests = {"S2 one drug unseen, 1:1": with_negatives(s2, all_pos),
             "S1 both drugs unseen, 1:1": with_negatives(s1, all_pos),
             f"S2 one drug unseen, 1:{real_ratio}": imbalanced(s2, all_pos, drugs, real_ratio, 3000, rng)}
    res["settings"]["cold_start"], _ = run_setting("cold", tr, tests, all_pos, fps, bits)

    # cold-aware training: hide one or both drugs' own interactions in every training pair
    g = build_graph(tr)
    ch = Chem2(fps, bits, g)
    train = with_negatives(tr, all_pos)
    hides = [(u,) if rng.random() < 0.5 else (u, v) for u, v in zip(train.u, train.v)]
    hides = [h if len(h) == 2 or rng.random() < 0.5 else (v,) for h, v in zip(hides, train.v)]
    X = np.stack([ch.pair_cold(u, v, h) for (u, v), h in zip(zip(train.u, train.v), hides)])
    m = fit(X, train.label.to_numpy())
    seen = set(tr.d1) | set(tr.d2)
    for tname, tdf in tests.items():
        Xt = np.stack([ch.pair_cold(u, v, tuple(d for d in (u, v) if d not in seen)) for u, v in zip(tdf.u, tdf.v)])
        r = scores(tdf.label.to_numpy(), m.predict_proba(Xt)[:, 1])
        res["settings"]["cold_start"][tname]["chem (cold-aware training)"] = r
        print(f"[cold-aware] {tname:28s} AUC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} F1 {r['f1']:.3f}")

    res["seconds"] = round(time.time() - t0)
    (ROOT / "docs/results/ddi_upgrades.json").write_text(json.dumps(res, indent=1))
    print(f"saved docs/results/ddi_upgrades.json ({res['seconds']}s)")


if __name__ == "__main__":
    main()
