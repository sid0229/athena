"""Paper / report figures from the saved results in docs/results (no recomputation).

    python scripts/make_figures.py

ddi_specificity.png      interaction judgement: LLM alone vs Athena's XGBoost predictor (research component)
f1_extraction.png        n2c2 micro F1 (strict and lenient) by extraction mode
f1_per_entity.png        n2c2 strict F1 per entity type: LLM only vs rules only vs hybrid
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from athena.config import ROOT  # noqa: E402

R = ROOT / "docs" / "results"
INK, MUTED = "#152238", "#50627A"
C = {"scispaCy only": "#A9B6C4", "LLM only": "#FF5A5F", "LLM only (plain prompt)": "#FF5A5F",
     "LLM only (cautious prompt)": "#FFA3A6", "Rules only": "#3B4F6B", "Athena hybrid": "#00A3B4",
     "XGBoost predictor (research)": "#00A3B4"}
plt.rcParams.update({"font.family": "Times New Roman", "axes.edgecolor": "#B8C4D0", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED})


def style(ax, title):
    ax.set_title(title, fontsize=12.5, color=INK, pad=10, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#E8EEF2", lw=0.8)
    ax.set_axisbelow(True)


def grouped(ax, groups, series, data, fmt="{:.2f}", ylim=(0, 1.1), tops=None):
    """tops[s][g]: y position for the value label (e.g. above an error bar)."""
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        vals = [data[s][g] for g in groups]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, label=s, color=C[s],
                      edgecolor="white", linewidth=0.6)
        for b, v, g in zip(bars, vals, groups):
            y = (tops or {}).get(s, {}).get(g, v)
            ax.text(b.get_x() + b.get_width() / 2, y + 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold" if s in ("Athena hybrid", "XGBoost predictor (research)") else "normal")
    ax.set_xticks(x, groups, fontsize=10.5)
    ax.set_ylim(*ylim)


def ddi_specificity():
    sp = json.loads((R / "specificity.json").read_text())["B_interaction_judgement"]
    res = sp["results"]
    order = ["LLM only (plain prompt)", "LLM only (cautious prompt)", "XGBoost predictor (research)"]
    rows = {k: res[k] for k in order if k in res}
    groups = ["Specificity", "Sensitivity", "Balanced accuracy"]
    data = {k: {"Specificity": v["specificity"], "Sensitivity": v["sensitivity"],
                "Balanced accuracy": v["balanced_accuracy"]} for k, v in rows.items()}
    tops = {k: {"Specificity": v["specificity_ci95"][1], "Sensitivity": v["sensitivity_ci95"][1]} for k, v in rows.items()}
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    grouped(ax, groups, list(rows), data, tops=tops, ylim=(0, 1.15))
    # 95% Wilson intervals (n = 100 per class)
    w = 0.8 / len(rows)
    for i, (k, v) in enumerate(rows.items()):
        for gi, (g, key) in enumerate((("Specificity", "specificity"), ("Sensitivity", "sensitivity"))):
            lo, hi = v[f"{key}_ci95"]
            ax.errorbar(gi + (i - (len(rows) - 1) / 2) * w, v[key], yerr=[[v[key] - lo], [hi - v[key]]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    ax.axhline(0.5, color="#9AA9B8", lw=0.9, ls="--")
    ax.text(2.42, 0.515, "chance", fontsize=8, color=MUTED, ha="right")
    style(ax, "Should the LLM judge drug–drug interactions? LLM alone vs a verified predictor")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.text(0.01, 0.012,
             f"{sp['pairs']} held-out DrugBank benchmark pairs, 50% interacting (frozen in {sp.get('pairs_file', 'specificity_pairs.json')}); "
             "error bars: 95% Wilson intervals. LLM = Llama 3.2 3B asked directly.\n"
             "XGBoost = Athena's research predictor (graph + chemistry, trained on training pairs only). Athena's live flags come "
             "from the DrugBank/DDInter lookup, not this model.\n"
             f"Benchmark negatives are sampled pairs: {sp.get('negatives_that_are_kb_interactions', '?')} of 100 are interactions "
             "in Athena's KB (results without them are in specificity.json).", fontsize=6.6, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(R / "ddi_specificity.png")
    plt.close(fig)


def _extraction_results():
    """The full 202-note test run if present, else the 35-note subset."""
    for name, label in (("extraction_test_202notes.json", "all 202 n2c2 2018 test notes"),
                        ("extraction_test_35notes.json", "first 35 of 202 n2c2 test notes")):
        if (R / name).exists():
            return json.loads((R / name).read_text()), label
    raise FileNotFoundError("no extraction results in docs/results")


def f1_modes():
    full, label = _extraction_results()
    d, boot, paired = full["results"], full.get("bootstrap", {}), full.get("paired_hybrid_vs_rules", {})
    names = {"scispacy": "scispaCy only", "llm": "LLM only", "rules": "Rules only", "hybrid": "Athena hybrid"}
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    x = np.arange(len(names))
    for kind, alpha, off in (("strict", 1.0, -0.19), ("lenient", 0.45, 0.19)):
        vals = [d[m][kind]["micro"]["f1"] for m in names]
        bars = ax.bar(x + off, vals, 0.36, color=[C[n] for n in names.values()], alpha=alpha, edgecolor="white")
        for b, v, m in zip(bars, vals, names):
            ci = boot.get(m, {}).get(f"{kind}_micro_f1_ci95")
            top = v
            if ci:
                ax.errorbar(b.get_x() + b.get_width() / 2, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none",
                            ecolor=INK, elinewidth=0.8, capsize=2.5)
                top = ci[1]
            ax.text(b.get_x() + b.get_width() / 2, top + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK, fontweight="bold" if m == "hybrid" else "normal")
    if paired.get("strict"):
        ps = paired["strict"]
        ax.text(0.99, 0.97, f"Hybrid − rules (strict): {ps['diff']:+.3f}, 95% CI [{ps['ci95'][0]:+.3f}, {ps['ci95'][1]:+.3f}]"
                + (" — significant" if ps["ci95"][0] > 0 else " — not significant"),
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
    ax.set_xticks(x, list(names.values()), fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Micro F1 (7 entity types)")
    style(ax, "Medication extraction F1 on held-out n2c2 2018 notes")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#5A6B80", label="Strict (exact span)"),
                       Patch(color="#5A6B80", alpha=0.45, label="Lenient (overlapping span)")],
              frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9.5)
    fig.text(0.01, 0.012, f"{label[0].upper() + label[1:]}; every mode on the same notes; tuning only on a separate validation "
             "split. Error bars: 95% bootstrap CIs over notes (2,000 resamples).", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(R / "f1_extraction.png")
    plt.close(fig)


def f1_entities():
    full, label = _extraction_results()
    d = full["results"]
    types = ["Drug", "Strength", "Route", "Dosage", "Form", "Frequency", "Duration"]
    series = {"LLM only": "llm", "Rules only": "rules", "Athena hybrid": "hybrid"}
    data = {s: {t: d[m]["strict"][t]["f1"] for t in types} for s, m in series.items()}
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)
    grouped(ax, types, list(series), data)
    ax.set_ylabel("Strict F1")
    style(ax, f"Extraction F1 per entity type ({label})")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9.5)
    fig.tight_layout()
    fig.savefig(R / "f1_per_entity.png")
    plt.close(fig)


if __name__ == "__main__":
    ddi_specificity()
    f1_modes()
    f1_entities()
    old = R / "specificity_extraction.png"
    if old.exists():
        old.unlink()
    print("saved docs/results/ddi_specificity.png, f1_extraction.png, f1_per_entity.png")
