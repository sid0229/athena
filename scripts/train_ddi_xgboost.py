"""Train and evaluate the XGBoost interaction predictor (Branch B, ML component).

    python scripts/train_ddi_xgboost.py            # all folds (~25 min on the M2)
    python scripts/train_ddi_xgboost.py --folds 1  # fold 0 only

Warm start (both drugs seen in training, new pairs), folds 0-2: graph + chemical-structure XGBoost
vs graph-only XGBoost, logistic regression and a drug-popularity baseline (degree product).
Cold start (one / both drugs unseen), folds 0-2: cold-aware chemistry model vs graph-only.

Model selection never touches test data: early stopping uses 10% of the training pairs (warm) or the
benchmark's val.csv, whose drugs are unseen like the test drugs (cold); the reported decision
threshold is chosen on that validation set. Metrics: ROC-AUC / PR-AUC with 95% bootstrap CIs on
fold 0, mean ± SD over folds. Writes docs/results/ddi_xgboost.json and ddi_xgboost_confusion.png and
saves the fold-0 models to data/processed/.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from athena.config import ROOT, data_path  # noqa: E402
from athena.data.drugbank import load_benchmark_ddi  # noqa: E402
from athena.verification.ddi_predictor import (CHEM_FEATURES, FEATURES, ChemFeatures, build_graph,  # noqa: E402
                                               featurize, load_fingerprints, with_negatives)

SEED = 42
LABELS = ["no interaction", "interaction"]
BASE = data_path("drugbank_benchmark")
XGB_PARAMS = dict(n_estimators=2000, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                  eval_metric="logloss", early_stopping_rounds=50, n_jobs=4, random_state=SEED, tree_method="hist")


def metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "prevalence": round(float(np.mean(y)), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "pr_auc": round(average_precision_score(y, prob), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def bootstrap_ci(y, prob, fn, n=200, seed=SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], prob[i]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def best_f1_threshold(y, prob) -> float:
    p, r, t = precision_recall_curve(y, prob)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return float(t[np.argmax(f[:-1])])


def fit_xgb(X, y, X_val, y_val):
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    return m


def corrupted_negatives(pos: pd.DataFrame, all_pos, drugs, ratio, n_pos, rng) -> pd.DataFrame:
    """n_pos positives + `ratio` distinct negatives each, made like the benchmark's (one side replaced)."""
    pos = pos.sample(min(n_pos, len(pos)), random_state=SEED)
    rows, seen = [(u, v, 1) for u, v in zip(pos.d1, pos.d2)], set()
    for u, v in zip(pos.d1, pos.d2):
        made = 0
        while made < ratio:
            w = rng.choice(drugs)
            a, b = (u, w) if rng.random() < 0.5 else (w, v)
            key = frozenset((a, b))
            if a != b and key not in all_pos and key not in seen:
                rows.append((a, b, 0))
                seen.add(key)
                made += 1
    return pd.DataFrame(rows, columns=["u", "v", "label"])


def warm_fold(fold, all_pos, fps, ratio, rng, full=False):
    t0 = time.time()
    train_df = pd.read_csv(BASE / f"warm_start/{fold}/train.csv")
    test_df = pd.read_csv(BASE / f"warm_start/{fold}/test.csv")
    # The benchmark lists a few unordered pairs in both splits; drop them from test.
    train_pairs = {frozenset(p) for p in zip(train_df.d1, train_df.d2)}
    keep = [frozenset(p) not in train_pairs for p in zip(test_df.d1, test_df.d2)]
    dropped = len(test_df) - sum(keep)
    test_df = test_df[keep]
    train, test = with_negatives(train_df, all_pos), with_negatives(test_df, all_pos)

    g = build_graph(train_df)  # training interactions only
    chem = ChemFeatures(fps, g)
    feats = lambda df: np.hstack([featurize(g, df), chem.featurize(df)]).astype(np.float32)  # noqa: E731
    X_tr, X_te = feats(train), feats(test)
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    ng = len(FEATURES)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr)
    del X_tr

    model = fit_xgb(X_fit, y_fit, X_val, y_val)
    graph_only = fit_xgb(X_fit[:, :ng], y_fit, X_val[:, :ng], y_val)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)).fit(X_fit[:, :ng], y_fit)
    thr = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])  # chosen on validation, never on test

    prob = model.predict_proba(X_te)[:, 1]
    prob_g = graph_only.predict_proba(X_te[:, :ng])[:, 1]
    popularity = X_te[:, FEATURES.index("pref_attachment")]  # degree product: "popular drugs interact"
    out = {
        "fold": fold, "train_pairs": int(len(train)), "test_pairs": int(len(test)),
        "test_pairs_removed_overlap": int(dropped), "best_iteration": int(model.best_iteration),
        "xgboost": metrics(y_te, prob), "xgboost_val_threshold": metrics(y_te, prob, thr),
        "xgboost_graph_only": metrics(y_te, prob_g),
        "baseline_logistic_regression": metrics(y_te, lr.predict_proba(X_te[:, :ng])[:, 1]),
        "baseline_popularity_auc": round(roc_auc_score(y_te, popularity), 4),
        "baseline_popularity_pr_auc": round(average_precision_score(y_te, popularity), 4),
    }
    print(f"[warm {fold}] AUC {out['xgboost']['roc_auc']} (graph-only {out['xgboost_graph_only']['roc_auc']}, "
          f"popularity {out['baseline_popularity_auc']}), best_iter {out['best_iteration']} ({time.time() - t0:.0f}s)",
          flush=True)
    if full:
        out["ci95"] = {"roc_auc": bootstrap_ci(y_te, prob, roc_auc_score),
                       "pr_auc": bootstrap_ci(y_te, prob, average_precision_score),
                       "graph_only_roc_auc": bootstrap_ci(y_te, prob_g, roc_auc_score)}
        seen = sorted(set(train_df.d1) | set(train_df.d2))
        out["imbalanced_test"] = {"note": f"test positives with 1:{ratio} (benchmark density) and 1:50 corrupted "
                                          "negatives, no duplicates; F1 at the validation-chosen threshold"}
        for name, df in ((f"1:{ratio}", corrupted_negatives(test_df, all_pos, seen, ratio, 4000, rng)),
                         ("1:50", corrupted_negatives(test_df, all_pos, seen, 50, 2000, rng))):
            Xi, yi = feats(df), df.label.to_numpy()
            out["imbalanced_test"][name] = {"xgboost": metrics(yi, model.predict_proba(Xi)[:, 1], thr),
                                            "xgboost_graph_only": metrics(yi, graph_only.predict_proba(Xi[:, :ng])[:, 1])}
        all_names = FEATURES + CHEM_FEATURES
        out["feature_importance_gain_top15"] = {all_names[int(k[1:])]: round(float(v), 3) for k, v in sorted(
            model.get_booster().get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:15]}
        return out, model, (y_te, prob)
    return out, None, None


def cold_fold(fold, all_pos, fps, rng):
    t0 = time.time()
    read = lambda n: (lambda d: d.assign(**{"Neg samples": d["split"]}))(  # noqa: E731
        pd.read_csv(BASE / f"cold_start/{fold}/{n}.csv"))  # cold files keep the negative in 'split'
    ctr, cval = read("train"), read("val")
    cg = build_graph(ctr)
    chem = ChemFeatures(fps, cg)
    seen = set(ctr.d1) | set(ctr.d2)
    unseen = lambda df: [tuple(d for d in (u, v) if d not in seen) for u, v in zip(df.u, df.v)]  # noqa: E731
    train, val = with_negatives(ctr, all_pos), with_negatives(cval, all_pos)
    # Hide one or both drugs' own interactions in every training pair, as at test time.
    hides = [(u, v) if rng.random() < 0.5 else ((u,) if rng.random() < 0.5 else (v,)) for u, v in zip(train.u, train.v)]
    X_val = chem.featurize(val, cold=True, hides=unseen(val))
    cold = fit_xgb(chem.featurize(train, cold=True, hides=hides), train.label.to_numpy(), X_val, val.label.to_numpy())
    graph = fit_xgb(featurize(cg, train), train.label.to_numpy(), featurize(cg, val), val.label.to_numpy())
    out = {"fold": fold, "best_iteration": int(cold.best_iteration)}
    for s in ("s2", "s1"):
        t = with_negatives(read(s), all_pos)
        y = t.label.to_numpy()
        prob = cold.predict_proba(chem.featurize(t, True, unseen(t)))[:, 1]
        out[s.upper()] = {"xgboost_cold_chemistry": metrics(y, prob),
                          "xgboost_graph_only": metrics(y, graph.predict_proba(featurize(cg, t))[:, 1]),
                          "n_pairs": int(len(t))}
        if fold == "fold0":
            out[s.upper()]["ci95_roc_auc"] = bootstrap_ci(y, prob, roc_auc_score)
    print(f"[cold {fold}] S2 {out['S2']['xgboost_cold_chemistry']['roc_auc']} S1 {out['S1']['xgboost_cold_chemistry']['roc_auc']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out, cold


def mean_sd(vals) -> dict:
    return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(SEED)
    bench = load_benchmark_ddi()
    all_pos = {frozenset(p) for p in zip(bench["id_a"], bench["id_b"])}
    n = len(set(bench.id_a) | set(bench.id_b))
    ratio = round((1 - (d := len(all_pos) / (n * (n - 1) / 2))) / d)  # benchmark density, ~1:7
    fps = load_fingerprints()
    folds = [f"fold{i}" for i in range(args.folds)]

    warm = []
    for f in folds:
        r, model, test_pred = warm_fold(f, all_pos, fps, ratio, rng, full=(f == "fold0"))
        warm.append(r)
        if model is not None:
            main_model, (y_te, prob), main = model, test_pred, r
    cold = []
    for f in folds:
        r, m = cold_fold(f, all_pos, fps, rng)
        cold.append(r)
        if f == "fold0":
            cold_model = m

    res = {
        "task": "binary DDI link prediction, DrugBank benchmark (DeepDDI); benchmark negatives (one drug replaced), "
                "negatives that are recorded interactions dropped",
        "features": FEATURES + ["chemical: Morgan r=2 256-bit sum/product, tanimoto, similarity to known partners, kNN links"],
        **{k: main[k] for k in ("train_pairs", "test_pairs", "test_pairs_removed_overlap", "best_iteration", "xgboost",
                                "xgboost_val_threshold", "xgboost_graph_only", "baseline_logistic_regression",
                                "baseline_popularity_auc", "baseline_popularity_pr_auc", "ci95", "imbalanced_test",
                                "feature_importance_gain_top15")},
        "warm_folds": warm,
        "warm_summary": {k: mean_sd([w[m]["roc_auc"] if m != "baseline_popularity_auc" else w[m] for w in warm])
                         for k, m in (("xgboost_roc_auc", "xgboost"), ("graph_only_roc_auc", "xgboost_graph_only"),
                                      ("popularity_roc_auc", "baseline_popularity_auc"))},
        "cold_start": {"note": "S2 = one drug unseen in training, S1 = both unseen; early stopping on val.csv; "
                               "1:1 benchmark negatives", **{k: cold[0][k] for k in ("S2", "S1")}},
        "cold_folds": cold,
        "cold_summary": {f"{s}_roc_auc": mean_sd([c[s]["xgboost_cold_chemistry"]["roc_auc"] for c in cold]) for s in ("S2", "S1")},
    }
    res["warm_summary"]["xgboost_pr_auc"] = mean_sd([w["xgboost"]["pr_auc"] for w in warm])
    res["seconds"] = round(time.time() - t0, 1)

    x, go, b = res["xgboost"], res["xgboost_graph_only"], res["baseline_logistic_regression"]
    pred = (prob >= 0.5).astype(int)
    print("\n=== XGBoost interaction predictor (graph + chemistry) — held-out test (warm-start fold 0) ===")
    print(classification_report(y_te, pred, target_names=LABELS, digits=3))
    print(f"ROC-AUC {x['roc_auc']} {res['ci95']['roc_auc']} · PR-AUC {x['pr_auc']} · test prevalence {x['prevalence']}")
    print(f"Popularity baseline AUC {res['baseline_popularity_auc']} · LR {b['roc_auc']} · graph-only {go['roc_auc']}")
    print("warm summary", res["warm_summary"], "\ncold summary", res["cold_summary"])

    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ddi_xgboost.json").write_text(json.dumps(res, indent=1))
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=160)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#152238", fontsize=11)
    ax.set_xticks([0, 1], ["No interaction", "Interaction"])
    ax.set_yticks([0, 1], ["No interaction", "Interaction"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"XGBoost DDI predictor — test set (fold 0)\nF1 {x['f1']:.3f} · ROC-AUC {x['roc_auc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ddi_xgboost_confusion.png")
    main_model.save_model(str(data_path("processed") / "ddi_xgboost.json"))
    cold_model.save_model(str(data_path("processed") / "ddi_xgboost_cold.json"))
    print(f"\nsaved docs/results/ddi_xgboost.json, docs/results/ddi_xgboost_confusion.png ({res['seconds']}s total)")


if __name__ == "__main__":
    main()
