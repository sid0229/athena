"""Compare the default multiplicative risk formula with the uncertainty-penalty variant.

    python scripts/compare_risk_formulas.py

Runs the full pipeline on the 35 cached n2c2 test notes and the 3 synthetic demo
patients with each formula, and reports how interaction findings are tiered.
Writes docs/results/risk_formula_comparison.json and ..._tiers.png.
"""

import copy
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402
from athena.fusion.report import analyze  # noqa: E402
from athena.verification.checker import InteractionChecker  # noqa: E402

FORMULAS = ["multiplicative", "uncertainty_penalty"]
TIERS = ["critical", "review", "info"]


def notes():
    cfg = load_config()["extraction"]
    ex = MedicationExtractor("hybrid")
    out = []
    for n in n2c2.load_split("test")[:40]:
        if all(ex.llm._cache_path(c.text).exists() for c in chunk_text(n.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"])):
            out.append(("n2c2:" + n.id, n.text))
    for p in sorted((ROOT / "data" / "demo").glob("*.txt")):
        out.append(("demo:" + p.stem, p.read_text()))
    return out, ex


def main():
    docs, ex = notes()
    checker = InteractionChecker()
    base = load_config()["fusion"]
    results = {f: {"tiers": Counter(), "by_severity": {}, "per_note_actionable": []} for f in FORMULAS}
    moves = Counter()
    for name, text in docs:
        per = {}
        for f in FORMULAS:
            cfg = copy.deepcopy(base)
            cfg["formula"] = f
            r = analyze(text, ex, checker, fusion_cfg=cfg)
            ints = [x for x in r.findings if x.type == "INTERACTION"]
            per[f] = {x.id: x.tier for x in ints}
            results[f]["tiers"].update(x.tier for x in ints)
            for x in ints:
                sev = x.evidence["severity"]
                results[f]["by_severity"].setdefault(sev, Counter())[x.tier] += 1
            results[f]["per_note_actionable"].append(sum(x.tier in ("critical", "review") for x in ints))
        for fid, t0 in per["multiplicative"].items():
            moves[(t0, per["uncertainty_penalty"].get(fid))] += 1

    n = len(docs)
    summary = {"notes": n, "formulas": {}}
    for f in FORMULAS:
        tot = sum(results[f]["tiers"].values())
        act = results[f]["per_note_actionable"]
        summary["formulas"][f] = {
            "interaction_findings": tot,
            "tiers": {t: results[f]["tiers"][t] for t in TIERS},
            "by_severity": {s: {t: c[t] for t in TIERS} for s, c in sorted(results[f]["by_severity"].items())},
            "mean_actionable_per_note": round(sum(act) / n, 1),
        }
    summary["tier_moves_default_to_penalty"] = {f"{a} -> {b}": c for (a, b), c in sorted(moves.items()) if a != b}

    print(f"{n} notes (n2c2 test + synthetic demo)\n")
    for f in FORMULAS:
        s = summary["formulas"][f]
        print(f"{f:22s} critical {s['tiers']['critical']:4d} | review {s['tiers']['review']:4d} | "
              f"info {s['tiers']['info']:4d} | actionable per note {s['mean_actionable_per_note']}")
        for sev, c in s["by_severity"].items():
            print(f"    {sev:9s} {c}")
    print("\nTier changes (default -> uncertainty penalty):", summary["tier_moves_default_to_penalty"])

    out = ROOT / "docs" / "results"
    (out / "risk_formula_comparison.json").write_text(json.dumps(summary, indent=1))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    colors = {"critical": "#C2413B", "review": "#D4930F", "info": "#2B7A8C"}
    labels = ["Multiplicative\n(current)", "Uncertainty penalty\n(proposed)"]
    left = [0, 0]
    for t in TIERS:
        vals = [summary["formulas"][f]["tiers"][t] for f in FORMULAS]
        ax.barh(labels, vals, left=left, color=colors[t], label=t.capitalize(), height=0.55)
        for i, v in enumerate(vals):
            if v:
                ax.text(left[i] + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlabel(f"Interaction findings across {n} notes")
    ax.set_title("How each risk formula tiers the same interactions", fontsize=11)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "risk_formula_tiers.png")
    print("\nsaved docs/results/risk_formula_comparison.json and risk_formula_tiers.png")


if __name__ == "__main__":
    main()
