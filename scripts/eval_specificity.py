"""Specificity experiments for the paper.

    python scripts/eval_specificity.py            # both parts (part B needs Ollama running)

A. Drug detection (n2c2 2018 test, cached notes): scispaCy-only vs LLM-only vs Athena hybrid,
   scored per token (is this token part of a Drug mention?) so true negatives exist and
   specificity = TN / (TN + FP) is defined.
B. Interaction judgement (DrugBank DDI benchmark, held-out pairs): the local LLM asked
   directly "do these drugs interact?" vs the XGBoost graph model, on the same balanced
   sample of interacting and non-interacting pairs.

Writes docs/results/specificity.json; figures are drawn by scripts/make_figures.py.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import ollama  # noqa: E402
import xgboost as xgb  # noqa: E402

from athena.config import ROOT, data_path, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.data.drugbank import load_benchmark_ddi  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.normalize.lexicon import load_lexicon  # noqa: E402
from athena.verification.ddi_predictor import (ChemFeatures, build_graph, featurize, load_fingerprints,  # noqa: E402
                                               load_split, with_negatives)

OUT = ROOT / "docs" / "results"
COL = {"scispaCy only": "#9AA9B8", "LLM only": "#FF5A5F", "Athena hybrid": "#00A3B4", "XGBoost (graph ML)": "#00A3B4"}
TOKEN = re.compile(r"\S+")


def rates(tp, fp, tn, fn):
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"sensitivity": round(sens, 4), "specificity": round(spec, 4), "precision": round(prec, 4),
            "f1": round(f1, 4), "fp_rate": round(1 - spec, 5), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ------------------------------------------------------------------ part A
def part_a():
    cfg = load_config()["extraction"]
    llm = OllamaExtractor()
    notes = [n for n in n2c2.load_split("test")[:40]
             if all(llm._cache_path(c.text).exists()
                    for c in chunk_text(n.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]))]
    modes = {"scispaCy only": "scispacy", "LLM only": "llm", "Athena hybrid": "hybrid"}
    res = {}
    for label, mode in modes.items():
        ex = MedicationExtractor(mode, llm=llm)
        tp = fp = tn = fn = 0
        for n in notes:
            gold = [(e.start, e.end) for e in n.drugs()]
            pred = [(m.drug.start, m.drug.end) for m in ex.extract(n.text).mentions]
            for t in TOKEN.finditer(n.text):
                s, e = t.span()
                g = any(a < e and s < b for a, b in gold)
                p = any(a < e and s < b for a, b in pred)
                tp += g and p
                fp += (not g) and p
                tn += (not g) and (not p)
                fn += g and (not p)
        res[label] = rates(tp, fp, tn, fn)
        print(f"A  {label:14s} {res[label]}")
    return {"notes": len(notes), "unit": "token", "entity": "Drug", "results": res}


# ------------------------------------------------------------------ part B
def part_b(n_per_class=100, seed=42, xgb_only=False):
    train_df, test_df = load_split("warm_start", "fold0")
    bench = load_benchmark_ddi()
    all_pos = {frozenset(p) for p in zip(bench["id_a"], bench["id_b"])}
    train_pairs = {frozenset(p) for p in zip(train_df.d1, train_df.d2)}
    test_df = test_df[[frozenset(p) not in train_pairs for p in zip(test_df.d1, test_df.d2)]]
    test = with_negatives(test_df, all_pos)
    id2name = {i: n for n, i in load_lexicon()["kaggle_name_to_id"].items()}
    test = test[test.u.isin(id2name) & test.v.isin(id2name)]
    rng = random.Random(seed)
    sample = (test[test.label == 1].sample(n_per_class, random_state=seed).to_dict("records")
              + test[test.label == 0].sample(n_per_class, random_state=seed).to_dict("records"))
    rng.shuffle(sample)

    # XGBoost: same model setup as scripts/train_ddi_xgboost.py (saved model)
    g = build_graph(train_df)
    model = xgb.XGBClassifier()
    model.load_model(str(data_path("processed") / "ddi_xgboost.json"))
    import pandas as pd
    sdf = pd.DataFrame(sample)
    X = np.hstack([featurize(g, sdf), ChemFeatures(load_fingerprints(), g).featurize(sdf)])
    xgb_pred = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    y = np.array([r["label"] for r in sample])
    if xgb_only:
        return xgb_pred, y

    # LLM asked directly, deterministic, constrained to yes/no
    llmcfg = load_config()["llm"]
    client = ollama.Client(host=llmcfg["host"], timeout=120)
    schema = {"type": "object", "properties": {"interaction": {"type": "string", "enum": ["yes", "no"]}},
              "required": ["interaction"]}
    llm_pred, t0 = [], time.time()
    for k, row in enumerate(sample):
        a, b = id2name[row["u"]], id2name[row["v"]]
        resp = client.chat(model=llmcfg["model"], format=schema,
                           options={"temperature": 0, "seed": 42, "num_ctx": 512, "num_predict": 20},
                           messages=[{"role": "system", "content": "You are a clinical pharmacology assistant. "
                                      "Answer only with yes or no."},
                                     {"role": "user", "content": f"Is there a known drug-drug interaction between "
                                      f"{a} and {b}?"}])
        llm_pred.append(1 if json.loads(resp.message.content)["interaction"] == "yes" else 0)
        if (k + 1) % 50 == 0:
            print(f"B  LLM judged {k + 1}/{len(sample)} pairs ({time.time() - t0:.0f}s)")

    y = np.array([r["label"] for r in sample])
    res = {}
    for label, pred in (("LLM only", np.array(llm_pred)), ("XGBoost (graph ML)", xgb_pred)):
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        res[label] = rates(tp, fp, tn, fn)
        res[label]["said_yes_pct"] = round(100 * pred.mean(), 1)
        print(f"B  {label:18s} {res[label]}")
    return {"pairs": len(sample), "positives": int(y.sum()), "model": llmcfg["model"],
            "reference": "DrugBank benchmark labels (recorded interaction vs benchmark negative)", "results": res}


# ------------------------------------------------------------------ plots
def bars(data, metrics, title, fname, ylim=(0, 1.05), note=None):
    labels = list(data)
    x = np.arange(len(metrics))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=170)
    for i, lab in enumerate(labels):
        vals = [data[lab][m] for m in metrics]
        bx = ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, w * 0.92, label=lab, color=COL[lab])
        for rect, v in zip(bx, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color="#50627A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / fname)
    plt.close(fig)


def main():
    if "--xgb-only" in sys.argv:  # refresh the XGBoost entry only; keep the saved LLM results
        out = json.loads((OUT / "specificity.json").read_text())
        pred, y = part_b(xgb_only=True)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        r = rates(tp, fp, tn, fn)
        r["said_yes_pct"] = round(100 * pred.mean(), 1)
        out["B_interaction_judgement"]["results"]["XGBoost (graph ML)"] = r
        print("B  XGBoost", r)
        (OUT / "specificity.json").write_text(json.dumps(out, indent=1))
        return
    out = {}
    a = part_a()
    out["A_drug_detection"] = a
    b = part_b()
    out["B_interaction_judgement"] = b
    (OUT / "specificity.json").write_text(json.dumps(out, indent=1))
    print("saved docs/results/specificity.json — draw figures with: python scripts/make_figures.py")


if __name__ == "__main__":
    main()
