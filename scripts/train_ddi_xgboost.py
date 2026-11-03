"""Train and evaluate the XGBoost interaction predictor (Branch B, ML component).

    python scripts/train_ddi_xgboost.py

Warm model (graph + chemical-structure features) on warm-start fold 0, compared with the
graph-only model and a logistic-regression baseline, at 1:1 and at realistic class ratios.
Cold model (chemical features, cold-aware training) on cold-start fold 0 (S2: one drug
unseen, S1: both unseen). Writes docs/results/ddi_xgboost.json and ddi_xgboost_confusion.png,
and saves both models to data/processed/.
"""

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
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
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


def metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "pr_auc": round(average_precision_score(y, prob), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def fit_xgb(X, y):
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", early_stopping_rounds=40, n_jobs=4, random_state=SEED, tree_method="hist")
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def corrupted_negatives(pos: pd.DataFrame, all_pos, drugs, ratio, n_pos, rng) -> pd.DataFrame:
    """n_pos positives + `ratio` negatives each, made like the benchmark's (one side replaced)."""
    pos = pos.sample(min(n_pos, len(pos)), random_state=SEED)
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


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    bench = load_benchmark_ddi()
    all_pos = {frozenset(p) for p in zip(bench["id_a"], bench["id_b"])}
    n = len(set(bench.id_a) | set(bench.id_b))
    ratio = round((1 - (d := len(all_pos) / (n * (n - 1) / 2))) / d)  # benchmark density, ~1:7
    fps = load_fingerprints()

    # ------------------------------------------------------------ warm start
    train_df = pd.read_csv(BASE / "warm_start/fold0/train.csv")
    test_df = pd.read_csv(BASE / "warm_start/fold0/test.csv")
    # The benchmark lists a few unordered pairs in both splits; drop them from test.
    train_pairs = {frozenset(p) for p in zip(train_df.d1, train_df.d2)}
    keep = [frozenset(p) not in train_pairs for p in zip(test_df.d1, test_df.d2)]
    dropped = len(test_df) - sum(keep)
    test_df = test_df[keep]
    train = with_negatives(train_df, all_pos)
    test = with_negatives(test_df, all_pos)
    seen = sorted(set(train_df.d1) | set(train_df.d2))
    imb = {f"1:{ratio}": corrupted_negatives(test_df, all_pos, seen, ratio, 4000, rng),
           "1:50": corrupted_negatives(test_df, all_pos, seen, 50, 2000, rng)}
    print(f"train pairs {len(train):,} (pos {train.label.sum():,}) | test pairs {len(test):,} "
          f"(pos {test.label.sum():,}) | test pairs also in train, removed: {dropped}")

    g = build_graph(train_df)  # training interactions only
    chem = ChemFeatures(fps, g)
    feats = lambda df: np.hstack([featurize(g, df), chem.featurize(df)])  # noqa: E731
    X_tr, X_te = feats(train), feats(test)
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    ng = len(FEATURES)
    print(f"features {X_tr.shape[1]} built in {time.time() - t0:.0f}s")

    model = fit_xgb(X_tr, y_tr)
    graph_only = fit_xgb(X_tr[:, :ng], y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    base.fit(X_tr[:, :ng], y_tr)

    print("\n=== XGBoost interaction predictor (graph + chemistry) — held-out test (warm-start fold 0) ===")
    print(classification_report(y_te, pred, target_names=LABELS, digits=3))
    cm = confusion_matrix(y_te, pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"                    pred: no   pred: yes\n  actual: no      {cm[0, 0]:>9,} {cm[0, 1]:>11,}"
          f"\n  actual: yes     {cm[1, 0]:>9,} {cm[1, 1]:>11,}")

    imbalanced = {}
    for name, df in imb.items():
        Xi = feats(df)
        yi = df.label.to_numpy()
        imbalanced[name] = {"xgboost": metrics(yi, model.predict_proba(Xi)[:, 1]),
                            "xgboost_graph_only": metrics(yi, graph_only.predict_proba(Xi[:, :ng])[:, 1])}
    all_names = FEATURES + CHEM_FEATURES
    res = {
        "task": "binary DDI link prediction, DrugBank benchmark (DeepDDI), warm-start fold0, 1:1 benchmark negatives",
        "features": FEATURES + ["chemical: Morgan r=2 256-bit sum/product, tanimoto, similarity to known partners, kNN links"],
        "train_pairs": int(len(train)), "test_pairs": int(len(test)),
        "best_iteration": int(model.best_iteration),
        "xgboost": metrics(y_te, prob),
        "xgboost_graph_only": metrics(y_te, graph_only.predict_proba(X_te[:, :ng])[:, 1]),
        "baseline_logistic_regression": metrics(y_te, base.predict_proba(X_te[:, :ng])[:, 1]),
        "imbalanced_test": {"note": f"test positives with 1:{ratio} (benchmark density) and 1:50 corrupted negatives",
                            **imbalanced},
        "feature_importance_gain_top15": {all_names[int(k[1:])]: round(float(v), 3) for k, v in sorted(
            model.get_booster().get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:15]},
        "test_pairs_removed_overlap": int(dropped),
    }
    x, go, b = res["xgboost"], res["xgboost_graph_only"], res["baseline_logistic_regression"]
    print(f"\nROC-AUC {x['roc_auc']} · PR-AUC {x['pr_auc']}")
    print(f"Logistic regression (graph): F1 {b['f1']} · ROC-AUC {b['roc_auc']}   |   XGBoost graph only: F1 {go['f1']} · "
          f"ROC-AUC {go['roc_auc']}   |   XGBoost graph + chemistry: F1 {x['f1']} · ROC-AUC {x['roc_auc']}")
    for name, r in imbalanced.items():
        print(f"{name}: PR-AUC graph only {r['xgboost_graph_only']['pr_auc']} -> graph + chemistry {r['xgboost']['pr_auc']}")

    # ------------------------------------------------------------ cold start
    ctr = pd.read_csv(BASE / "cold_start/fold0/train.csv")
    ctr["Neg samples"] = ctr["split"]  # cold-start files keep the negative in the 'split' column
    cg = build_graph(ctr)
    cchem = ChemFeatures(fps, cg)
    ctrain = with_negatives(ctr, all_pos)
    # Hide one or both drugs' own interactions in every training pair, as at test time.
    hides = [(u, v) if rng.random() < 0.5 else ((u,) if rng.random() < 0.5 else (v,))
             for u, v in zip(ctrain.u, ctrain.v)]
    cold = fit_xgb(cchem.featurize(ctrain, cold=True, hides=hides), ctrain.label.to_numpy())
    cold_graph = fit_xgb(featurize(cg, ctrain), ctrain.label.to_numpy())
    cseen = set(ctr.d1) | set(ctr.d2)
    res["cold_start"] = {"note": "fold0; S2 = one drug unseen in training, S1 = both unseen; 1:1 benchmark negatives"}
    for s in ("s2", "s1"):
        df = pd.read_csv(BASE / f"cold_start/fold0/{s}.csv")
        df["Neg samples"] = df["split"]
        t = with_negatives(df, all_pos)
        hid = [tuple(d for d in (u, v) if d not in cseen) for u, v in zip(t.u, t.v)]
        res["cold_start"][s.upper()] = {
            "xgboost_cold_chemistry": metrics(t.label.to_numpy(), cold.predict_proba(cchem.featurize(t, True, hid))[:, 1]),
            "xgboost_graph_only": metrics(t.label.to_numpy(), cold_graph.predict_proba(featurize(cg, t))[:, 1])}
        r = res["cold_start"][s.upper()]
        print(f"cold {s.upper()}: ROC-AUC graph only {r['xgboost_graph_only']['roc_auc']} -> "
              f"cold-aware chemistry {r['xgboost_cold_chemistry']['roc_auc']}")
    res["seconds"] = round(time.time() - t0, 1)

    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ddi_xgboost.json").write_text(json.dumps(res, indent=1))

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
    ax.set_title(f"XGBoost DDI predictor — test set\nF1 {x['f1']:.3f} · ROC-AUC {x['roc_auc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ddi_xgboost_confusion.png")
    model.save_model(str(data_path("processed") / "ddi_xgboost.json"))
    cold.save_model(str(data_path("processed") / "ddi_xgboost_cold.json"))
    print(f"\nsaved docs/results/ddi_xgboost.json, docs/results/ddi_xgboost_confusion.png "
          f"({res['seconds']}s total)")


if __name__ == "__main__":
    main()
