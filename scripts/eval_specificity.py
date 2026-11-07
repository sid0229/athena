"""Specificity experiments for the paper.

    python scripts/eval_specificity.py            # both parts (part B needs Ollama running)

A. Drug detection (n2c2 2018 test, cached notes): scispaCy-only vs LLM-only vs Athena hybrid,
   scored per token (is this token part of a Drug mention?) so true negatives exist and
   specificity = TN / (TN + FP) is defined.
B. Interaction judgement (DrugBank DDI benchmark, held-out pairs): the local LLM asked
   directly "do these drugs interact?" (plain and cautious prompts) vs the XGBoost predictor
   (research component), on the same frozen balanced sample of 200 pairs
   (docs/results/specificity_pairs.json). Benchmark negatives are sampled pairs, not confirmed
   non-interactions: those that are interactions in Athena's own KB are flagged, and every
   metric is also reported without them. 95% Wilson intervals on sensitivity / specificity.

    python scripts/eval_specificity.py --part b          # XGBoost only (no LLM)
    python scripts/eval_specificity.py --part b --llm    # + both LLM prompts (needs Ollama)

Writes docs/results/specificity.json; figures are drawn by scripts/make_figures.py.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import ollama  # noqa: E402
import pandas as pd  # noqa: E402
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
PAIRS = OUT / "specificity_pairs.json"
XGB = "XGBoost predictor (research)"
PROMPTS = {
    "LLM only (plain prompt)": (
        {"type": "object", "properties": {"interaction": {"type": "string", "enum": ["yes", "no"]}},
         "required": ["interaction"]},
        "You are a clinical pharmacology assistant. Answer only with yes or no.",
        "Is there a known drug-drug interaction between {a} and {b}?", "interaction", "yes"),
    "LLM only (cautious prompt)": (
        {"type": "object", "properties": {"answer": {"type": "string",
                                                     "enum": ["interaction", "no interaction", "not sure"]}},
         "required": ["answer"]},
        "You are a careful clinical pharmacist. Most random drug pairs do NOT interact. Only say 'interaction' "
        "if you are confident a documented interaction exists; otherwise say 'no interaction' or 'not sure'.",
        "Drug A: {a}\nDrug B: {b}\nIs there a documented drug-drug interaction?", "answer", "interaction"),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [round(c - h, 3), round(c + h, 3)]


def score(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    r = rates(tp, fp, tn, fn)
    r["balanced_accuracy"] = round((r["sensitivity"] + r["specificity"]) / 2, 4)
    r["sensitivity_ci95"], r["specificity_ci95"] = wilson(tp, tp + fn), wilson(tn, tn + fp)
    r["said_yes_pct"] = round(100 * pred.mean(), 1)
    return r


def sample_pairs(n_per_class=100, seed=42):
    """The frozen 200-pair sample (created once, then always reloaded)."""
    if PAIRS.exists():
        return json.loads(PAIRS.read_text())
    from athena.verification.checker import InteractionChecker
    train_df, test_df = load_split("warm_start", "fold0")
    bench = load_benchmark_ddi()
    all_pos = {frozenset(p) for p in zip(bench["id_a"], bench["id_b"])}
    train_pairs = {frozenset(p) for p in zip(train_df.d1, train_df.d2)}
    test_df = test_df[[frozenset(p) not in train_pairs for p in zip(test_df.d1, test_df.d2)]]
    test = with_negatives(test_df, all_pos)
    id2name = {i: n for n, i in load_lexicon()["kaggle_name_to_id"].items()}
    test = test[test.u.isin(id2name) & test.v.isin(id2name)]
    sample = (test[test.label == 1].sample(n_per_class, random_state=seed).to_dict("records")
              + test[test.label == 0].sample(n_per_class, random_state=seed).to_dict("records"))
    random.Random(seed).shuffle(sample)
    chk = InteractionChecker()
    for r in sample:
        r["name_u"], r["name_v"] = id2name[r["u"]], id2name[r["v"]]
        rep = chk.check([r["name_u"], r["name_v"]])
        r["kb_interaction"] = bool(rep.interactions)  # Athena's KB says these interact (DrugBank or DDInter)
        r["label"] = int(r["label"])
    PAIRS.write_text(json.dumps(sample, indent=1))
    return sample


def part_b(run_llm=False):
    sample = sample_pairs()
    sdf = pd.DataFrame(sample)
    y = sdf.label.to_numpy()
    clean = ~((sdf.label == 0) & sdf.kb_interaction).to_numpy()  # drop negatives our own KB knows interact
    preds = {}
    prev = json.loads((OUT / "specificity.json").read_text()).get("B_interaction_judgement", {}) \
        if (OUT / "specificity.json").exists() else {}
    for name, (schema, system, user, key, yes) in PROMPTS.items():
        if run_llm:
            llmcfg = load_config()["llm"]
            client = ollama.Client(host=llmcfg["host"], timeout=120)
            out = []
            for r in sample:
                resp = client.chat(model=llmcfg["model"], format=schema,
                                   options={"temperature": 0, "seed": 42, "num_ctx": 512, "num_predict": 20},
                                   messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": user.format(a=r["name_u"], b=r["name_v"])}])
                out.append(1 if json.loads(resp.message.content)[key] == yes else 0)
            preds[name] = np.array(out)
        elif name in prev.get("predictions", {}):
            preds[name] = np.array(prev["predictions"][name])
    g = build_graph(load_split("warm_start", "fold0")[0])
    model = xgb.XGBClassifier()
    model.load_model(str(data_path("processed") / "ddi_xgboost.json"))
    X = np.hstack([featurize(g, sdf), ChemFeatures(load_fingerprints(), g).featurize(sdf)])
    preds[XGB] = (model.predict_proba(X)[:, 1] >= 0.5).astype(int)
    res = {"pairs": len(sample), "positives": int(y.sum()),
           "negatives_that_are_kb_interactions": int(((sdf.label == 0) & sdf.kb_interaction).sum()),
           "model": load_config()["llm"]["model"], "pairs_file": PAIRS.name,
           "reference": "DrugBank benchmark labels (recorded interaction vs sampled benchmark negative)",
           "results": {n: score(p, y) for n, p in preds.items()},
           "results_excluding_kb_positive_negatives": {n: score(p[clean], y[clean]) for n, p in preds.items()},
           "predictions": {n: p.tolist() for n, p in preds.items()}}
    for n, r in res["results"].items():
        print(f"B  {n:28s} sens {r['sensitivity']:.2f} {r['sensitivity_ci95']} spec {r['specificity']:.2f} "
              f"{r['specificity_ci95']} bal.acc {r['balanced_accuracy']:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--llm", action="store_true", help="re-run both LLM prompts (needs Ollama)")
    args = ap.parse_args()
    path = OUT / "specificity.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if args.part in ("a", "both"):
        out["A_drug_detection"] = part_a()
    if args.part in ("b", "both"):
        out["B_interaction_judgement"] = part_b(args.llm or args.part == "both")
    path.write_text(json.dumps(out, indent=1))
    print("saved docs/results/specificity.json — draw figures with: python scripts/make_figures.py")


if __name__ == "__main__":
    main()
