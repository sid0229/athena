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
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from athena.config import ROOT, data_path  # noqa: E402
from athena.data.drugbank import load_benchmark_ddi  # noqa: E402
from athena.verification.ddi_predictor import build_graph, featurize, with_negatives  # noqa: E402

RDLogger.DisableLog("rdApp.*")
SEED, NBITS, K = 42, 256, 5
BASE = data_path("drugbank_benchmark")


# ------------------------------------------------------------------ chemistry
def load_fps():
    smi = pd.read_csv(BASE / "drug_smiles.csv")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    fps, bits = {}, {}
    for d, s in zip(smi.drug_id, smi.smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fp = gen.GetFingerprint(m)
        fps[d] = fp
        bits[d] = np.array(list(fp), dtype=np.uint8)
    return fps, bits


class Chem2:
    def __init__(self, fps, bits, g):
        self.fps, self.bits, self.g = fps, bits, g
        self.ids = list(fps)
        self.fplist = [fps[d] for d in self.ids]
        self._sim, self._knn = {}, {}
        self.known = [d for d in self.ids if g.get(d)]  # drugs with training interactions
        self.known_fps = [fps[d] for d in self.known]

    def sims(self, d):  # Tanimoto of d to every drug with SMILES
        if d not in self._sim:
            self._sim[d] = dict(zip(self.ids, DataStructs.BulkTanimotoSimilarity(self.fps[d], self.fplist)))
        return self._sim[d]

    def knn(self, d):  # K most similar drugs that have training interactions (excluding d)
        if d not in self._knn:
            s = DataStructs.BulkTanimotoSimilarity(self.fps[d], self.known_fps)
            order = np.argsort(s)[::-1]
            self._knn[d] = [self.known[i] for i in order if self.known[i] != d][:K]
        return self._knn[d]

    def pair(self, u, v):
        zero = np.zeros(2 * NBITS + 5, dtype=np.float32)
        if u not in self.fps or v not in self.fps:
            return zero
        bu, bv = self.bits[u], self.bits[v]
        su = self.sims(u)
        nu, nv = self.g.get(u, set()) - {v}, self.g.get(v, set()) - {u}
        max_u_to_nv = max((su[w] for w in nv if w in su), default=0.0)
        sv = self.sims(v)
        max_v_to_nu = max((sv[w] for w in nu if w in sv), default=0.0)
        ku, kv = self.knn(u), self.knn(v)
        hits = sum(1 for a in ku for b in kv if a != b and {a, b} != {u, v} and b in self.g.get(a, ()))
        return np.concatenate([bu + bv, bu * bv,
                               [su[v], max_u_to_nv, max_v_to_nu, max(max_u_to_nv, max_v_to_nu), hits / (K * K)]]
                              ).astype(np.float32)

    def featurize(self, pairs):
        return np.stack([self.pair(u, v) for u, v in zip(pairs["u"], pairs["v"])])

    def pair_cold(self, u, v, hide=()):
        """Features usable when a drug has no recorded interactions. In training, the drugs in
        `hide` are treated as unseen (their own interactions ignored), as they will be at test time."""
        if u not in self.fps or v not in self.fps:
            return np.zeros(2 * NBITS + 4, dtype=np.float32)
        nb = lambda d, other: set() if d in hide else self.g.get(d, set()) - {other}  # noqa: E731
        su, sv = self.sims(u), self.sims(v)
        a = max((su[w] for w in nb(v, u) if w in su), default=0.0)
        b = max((sv[w] for w in nb(u, v) if w in sv), default=0.0)
        ku = [x for x in self.knn(u) if x not in hide][:K]
        kv = [x for x in self.knn(v) if x not in hide][:K]
        hits = sum(1 for x in ku for y in kv if x != y and {x, y} != {u, v} and y in self.g.get(x, ()))
        return np.concatenate([self.bits[u] + self.bits[v], self.bits[u] * self.bits[v],
                               [su[v], max(a, b), min(a, b), hits / (K * K)]]).astype(np.float32)


# ------------------------------------------------------------------ data
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
