"""Train and evaluate the XGBoost interaction predictor (Branch B, ML component).

    python scripts/train_ddi_xgboost.py

Prints a classification report and confusion matrix, and writes
docs/results/ddi_xgboost.json and docs/results/ddi_xgboost_confusion.png.
A logistic-regression model on the same features is reported as a baseline.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from athena.config import ROOT, data_path  # noqa: E402
from athena.data.drugbank import load_benchmark_ddi  # noqa: E402
from athena.verification.ddi_predictor import (FEATURES, build_graph, featurize, load_split,  # noqa: E402
                                               with_negatives)

SEED = 42
LABELS = ["no interaction", "interaction"]


def metrics(y, prob, thr=0.5) -> dict:
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "pr_auc": round(average_precision_score(y, prob), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    t0 = time.time()
    train_df, test_df = load_split("warm_start", "fold0")
    all_pos = {frozenset(p) for p in zip(*[load_benchmark_ddi()[c] for c in ("id_a", "id_b")])}
    # The benchmark lists a few unordered pairs in both splits; drop them from test.
    train_pairs = {frozenset(p) for p in zip(train_df.d1, train_df.d2)}
    keep = [frozenset(p) not in train_pairs for p in zip(test_df.d1, test_df.d2)]
    dropped = len(test_df) - sum(keep)
    test_df = test_df[keep]
    train = with_negatives(train_df, all_pos)
    test = with_negatives(test_df, all_pos)
    print(f"train pairs {len(train):,} (pos {train.label.sum():,}) | test pairs {len(test):,} "
          f"(pos {test.label.sum():,}) | test pairs also in train, removed: {dropped}")

    g = build_graph(train_df)  # training interactions only
    X_tr, X_te = featurize(g, train), featurize(g, test)
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
    print(f"features {X_tr.shape[1]} built in {time.time() - t0:.0f}s")

    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr)
    model = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", early_stopping_rounds=40, n_jobs=4, random_state=SEED, tree_method="hist")
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    base.fit(X_tr, y_tr)
    base_prob = base.predict_proba(X_te)[:, 1]

    print("\n=== XGBoost interaction predictor — held-out test (warm-start fold 0) ===")
    print(classification_report(y_te, pred, target_names=LABELS, digits=3))
    cm = confusion_matrix(y_te, pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"                    pred: no   pred: yes\n  actual: no      {cm[0, 0]:>9,} {cm[0, 1]:>11,}"
          f"\n  actual: yes     {cm[1, 0]:>9,} {cm[1, 1]:>11,}")
    res = {
        "task": "binary DDI link prediction, DrugBank benchmark (DeepDDI), warm-start fold0, 1:1 benchmark negatives",
        "features": FEATURES,
        "train_pairs": int(len(train)), "test_pairs": int(len(test)),
        "best_iteration": int(model.best_iteration),
        "xgboost": metrics(y_te, prob),
        "baseline_logistic_regression": metrics(y_te, base_prob),
        "feature_importance_gain": {FEATURES[int(k[1:])]: round(float(v), 3) for k, v in sorted(
            model.get_booster().get_score(importance_type="gain").items(), key=lambda kv: -kv[1])},
        "test_pairs_removed_overlap": int(dropped),
        "seconds": round(time.time() - t0, 1),
    }
    b, x = res["baseline_logistic_regression"], res["xgboost"]
    print(f"\nROC-AUC {x['roc_auc']} · PR-AUC {x['pr_auc']}")
    print(f"Baseline logistic regression: F1 {b['f1']} · ROC-AUC {b['roc_auc']}   |   XGBoost: F1 {x['f1']} · ROC-AUC {x['roc_auc']}")

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
    print(f"\nsaved docs/results/ddi_xgboost.json, docs/results/ddi_xgboost_confusion.png "
          f"({res['seconds']}s total)")


if __name__ == "__main__":
    main()
