"""Paper / report figures from the saved results in docs/results (no recomputation).

    python scripts/make_figures.py

ddi_specificity.png      interaction judgement: LLM alone vs Athena hybrid verification
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
     "LLM only (cautious prompt)": "#FFA3A6", "Rules only": "#3B4F6B", "Athena hybrid": "#00A3B4"}
plt.rcParams.update({"font.family": "Times New Roman", "axes.edgecolor": "#B8C4D0", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED})


def style(ax, title):
    ax.set_title(title, fontsize=12.5, color=INK, pad=10, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#E8EEF2", lw=0.8)
    ax.set_axisbelow(True)


def grouped(ax, groups, series, data, fmt="{:.2f}", ylim=(0, 1.1)):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        vals = [data[s][g] for g in groups]
        bars = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.9, label=s, color=C[s],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold" if s == "Athena hybrid" else "normal")
    ax.set_xticks(x, groups, fontsize=10.5)
    ax.set_ylim(*ylim)


def ddi_specificity():
    sp = json.loads((R / "specificity.json").read_text())["B_interaction_judgement"]
    plain, careful, hyb = sp["results"]["LLM only"], sp["llm_cautious_prompt"], sp["results"]["XGBoost (graph ML)"]
    rows = {"LLM only (plain prompt)": plain, "LLM only (cautious prompt)": careful, "Athena hybrid": hyb}
    groups = ["Specificity", "Sensitivity", "Balanced accuracy"]
    data = {k: {"Specificity": v["specificity"], "Sensitivity": v["sensitivity"],
                "Balanced accuracy": (v["specificity"] + v["sensitivity"]) / 2} for k, v in rows.items()}
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=200)
    grouped(ax, groups, list(rows), data)
    ax.axhline(0.5, color="#9AA9B8", lw=0.9, ls="--")
    ax.text(2.42, 0.515, "chance", fontsize=8, color=MUTED, ha="right")
    style(ax, "Drug–drug interaction judgement: LLM alone vs Athena hybrid")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9.5)
    fig.text(0.01, 0.012, f"{sp['pairs']} held-out DrugBank benchmark pairs (50% interacting). LLM = Llama 3.2 3B asked directly; "
             "Athena hybrid = LLM restricted to extraction, interactions decided by the verification branch "
             "(XGBoost on interaction-graph + chemical-structure features, trained only on training pairs).", fontsize=7, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(R / "ddi_specificity.png")
    plt.close(fig)


def f1_modes():
    d = json.loads((R / "extraction_test_35notes.json").read_text())["results"]
    names = {"scispacy": "scispaCy only", "llm": "LLM only", "rules": "Rules only", "hybrid": "Athena hybrid"}
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    x = np.arange(len(names))
    for j, (kind, alpha, off) in enumerate((("strict", 1.0, -0.19), ("lenient", 0.45, 0.19))):
        vals = [d[m][kind]["micro"]["f1"] for m in names]
        bars = ax.bar(x + off, vals, 0.36, color=[C[n] for n in names.values()], alpha=alpha,
                      edgecolor="white", label=f"{kind.title()} match")
        for b, v, m in zip(bars, vals, names):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color=INK, fontweight="bold" if m == "hybrid" else "normal")
    ax.set_xticks(x, list(names.values()), fontsize=10.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Micro F1 (7 entity types)")
    style(ax, "Medication extraction F1 on held-out n2c2 2018 notes")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#5A6B80", label="Strict (exact span)"),
                       Patch(color="#5A6B80", alpha=0.45, label="Lenient (overlapping span)")],
              frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9.5)
    fig.text(0.01, 0.012, "First 35 of 202 n2c2 test notes; every mode scored on the same notes. Tuning was done only on a separate validation split.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(R / "f1_extraction.png")
    plt.close(fig)


def f1_entities():
    d = json.loads((R / "extraction_test_35notes.json").read_text())["results"]
    types = ["Drug", "Strength", "Route", "Dosage", "Form", "Frequency", "Duration"]
    series = {"LLM only": "llm", "Rules only": "rules", "Athena hybrid": "hybrid"}
    data = {s: {t: d[m]["strict"][t]["f1"] for t in types} for s, m in series.items()}
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)
    grouped(ax, types, list(series), data)
    ax.set_ylabel("Strict F1")
    style(ax, "Extraction F1 per entity type (held-out n2c2 notes)")
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
