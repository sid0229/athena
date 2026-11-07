"""Score Branch A on n2c2 2018 Track 2.

    python scripts/eval_extraction.py val                       # all modes, cached LLM output only
    python scripts/eval_extraction.py test --limit 40 --save    # writes docs/results/

Only notes whose LLM output is fully cached are scored for llm/hybrid unless
--run-llm is given (so scoring never competes with a running cache job).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.data.n2c2 import V1_TYPES  # noqa: E402
from athena.eval.n2c2_metrics import Scores, score_note  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402


def fully_cached(llm: OllamaExtractor, text: str, cfg: dict) -> bool:
    return all(llm._cache_path(c.text).exists()
               for c in chunk_text(text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]))


def table(name: str, s: Scores) -> str:
    lines = [f"\n== {name}", f"{'type':10s} {'strict P/R/F1':>22s}   {'lenient P/R/F1':>22s}"]
    for t in (*V1_TYPES, "micro"):
        a = s.micro("strict") if t == "micro" else s.strict[t]
        b = s.micro("lenient") if t == "micro" else s.lenient[t]
        lines.append(f"{t:10s} {a.p:6.3f} {a.r:6.3f} {a.f1:6.3f}   {b.p:6.3f} {b.r:6.3f} {b.f1:6.3f}")
    lines.append(f"{'relations':10s} {s.rel_strict.p:6.3f} {s.rel_strict.r:6.3f} {s.rel_strict.f1:6.3f}   "
                 f"{s.rel_lenient.p:6.3f} {s.rel_lenient.r:6.3f} {s.rel_lenient.f1:6.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--modes", default="scispacy,rules,llm,hybrid")
    ap.add_argument("--run-llm", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score only the fully cached notes (e.g. the 35-note subset) instead of stopping")
    args = ap.parse_args()

    cfg = load_config()
    notes = n2c2.load_split(args.split)[: args.limit]
    llm = OllamaExtractor()
    if not args.run_llm:
        # Cache-only scoring: never call the model and never fall back to rules silently
        # (MedicationExtractor falls back to rules when Ollama is not running).
        llm.available = lambda: True

        def _no_call(text):
            raise RuntimeError("LLM output not cached for a chunk — run with --run-llm")
        llm._call = _no_call
    if args.run_llm:
        eligible = list(notes)
    else:
        eligible = [n for n in notes if fully_cached(llm, n.text, cfg["extraction"])]
    print(f"{args.split}: {len(notes)} notes, {len(eligible)} with cached LLM output")
    if not args.run_llm and len(eligible) < len(notes) and not args.allow_partial:
        raise RuntimeError(
            f"Cache-only mode: only {len(eligible)}/{len(notes)} notes are fully cached. "
            f"Run scripts/warm_llm_cache.py first to fill any gaps."
        )

    results = {}
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    mode_note_scores = {m: [] for m in mode_names}

    for mode in mode_names:
        ex = MedicationExtractor(mode, llm=llm)
        use = eligible  # every mode on the same notes, so scores are comparable
        scores, t0, dropped, total = Scores(), time.time(), 0, 0
        for note in use:
            r = ex.extract(note.text)
            assert r.mode == mode, f"{note.id}: scored as {r.mode}, not {mode} ({r.warnings})"
            n_score = score_note(note, r.mentions)
            score_note(note, r.mentions, scores)

            sm = n_score.micro("strict")
            lm = n_score.micro("lenient")
            mode_note_scores[mode].append({
                "strict_tp": sm.tp, "strict_fp": sm.fp, "strict_fn": sm.fn,
                "lenient_tp": lm.tp, "lenient_fp": lm.fp, "lenient_fn": lm.fn,
            })
            if r.stats:
                dropped += len(r.stats.values_dropped)
                total += r.stats.values_total
        print(table(f"{mode} ({len(use)} notes, {time.time() - t0:.0f}s)", scores))
        if total:
            print(f"LLM attribute values not found in text (dropped): {dropped}/{total} ({dropped / total:.1%})")
        results[mode] = {"notes": [n.id for n in use], "n_notes": len(use), **scores.as_dict(),
                         "llm_values_dropped": dropped, "llm_values_total": total}

    # Bootstrap 95% confidence intervals (2000 resamples over notes)
    import numpy as np
    n_notes = len(eligible)
    rng = np.random.default_rng(42)
    B = 2000
    boot_indices = rng.choice(n_notes, size=(B, n_notes), replace=True)

    bootstrap_summary = {}
    mode_boot_samples = {}
    for mode in mode_names:
        note_stats = mode_note_scores[mode]
        s_tp = np.array([x["strict_tp"] for x in note_stats])
        s_fp = np.array([x["strict_fp"] for x in note_stats])
        s_fn = np.array([x["strict_fn"] for x in note_stats])

        l_tp = np.array([x["lenient_tp"] for x in note_stats])
        l_fp = np.array([x["lenient_fp"] for x in note_stats])
        l_fn = np.array([x["lenient_fn"] for x in note_stats])

        b_s_tp = s_tp[boot_indices].sum(axis=1)
        b_s_fp = s_fp[boot_indices].sum(axis=1)
        b_s_fn = s_fn[boot_indices].sum(axis=1)
        denom_s = 2 * b_s_tp + b_s_fp + b_s_fn
        strict_f1s = np.where(denom_s > 0, 2 * b_s_tp / denom_s, 0.0)

        b_l_tp = l_tp[boot_indices].sum(axis=1)
        b_l_fp = l_fp[boot_indices].sum(axis=1)
        b_l_fn = l_fn[boot_indices].sum(axis=1)
        denom_l = 2 * b_l_tp + b_l_fp + b_l_fn
        lenient_f1s = np.where(denom_l > 0, 2 * b_l_tp / denom_l, 0.0)

        mode_boot_samples[mode] = {"strict": strict_f1s, "lenient": lenient_f1s}
        bootstrap_summary[mode] = {
            "strict_micro_f1": results[mode]["strict"]["micro"]["f1"],
            "strict_micro_f1_ci95": [round(float(np.percentile(strict_f1s, 2.5)), 4),
                                     round(float(np.percentile(strict_f1s, 97.5)), 4)],
            "lenient_micro_f1": results[mode]["lenient"]["micro"]["f1"],
            "lenient_micro_f1_ci95": [round(float(np.percentile(lenient_f1s, 2.5)), 4),
                                      round(float(np.percentile(lenient_f1s, 97.5)), 4)],
        }

    paired_summary = {}
    if "hybrid" in mode_boot_samples and "rules" in mode_boot_samples:
        diff_strict = mode_boot_samples["hybrid"]["strict"] - mode_boot_samples["rules"]["strict"]
        diff_lenient = mode_boot_samples["hybrid"]["lenient"] - mode_boot_samples["rules"]["lenient"]
        h_s = results["hybrid"]["strict"]["micro"]["f1"]
        r_s = results["rules"]["strict"]["micro"]["f1"]
        h_l = results["hybrid"]["lenient"]["micro"]["f1"]
        r_l = results["rules"]["lenient"]["micro"]["f1"]
        paired_summary = {
            "strict": {
                "diff": round(h_s - r_s, 4),
                "ci95": [round(float(np.percentile(diff_strict, 2.5)), 4),
                         round(float(np.percentile(diff_strict, 97.5)), 4)],
            },
            "lenient": {
                "diff": round(h_l - r_l, 4),
                "ci95": [round(float(np.percentile(diff_lenient, 2.5)), 4),
                         round(float(np.percentile(diff_lenient, 97.5)), 4)],
            },
        }

    print("\n== 95% Bootstrap Confidence Intervals (2,000 resamples over notes)")
    print(f"{'Mode':12s} {'Strict Micro F1 (95% CI)':>30s} {'Lenient Micro F1 (95% CI)':>30s}")
    for mode in mode_names:
        bs = bootstrap_summary[mode]
        s_ci = f"{bs['strict_micro_f1']:.4f} [{bs['strict_micro_f1_ci95'][0]:.4f}, {bs['strict_micro_f1_ci95'][1]:.4f}]"
        l_ci = f"{bs['lenient_micro_f1']:.4f} [{bs['lenient_micro_f1_ci95'][0]:.4f}, {bs['lenient_micro_f1_ci95'][1]:.4f}]"
        print(f"{mode:12s} {s_ci:>30s} {l_ci:>30s}")

    if paired_summary:
        print("\n== Paired Difference: Hybrid - Rules")
        ps = paired_summary["strict"]
        pl = paired_summary["lenient"]
        print(f"Strict  diff: {ps['diff']:+.4f} (95% CI [{ps['ci95'][0]:+.4f}, {ps['ci95'][1]:+.4f}])")
        print(f"Lenient diff: {pl['diff']:+.4f} (95% CI [{pl['ci95'][0]:+.4f}, {pl['ci95'][1]:+.4f}])")

    if args.save:
        out = ROOT / "docs" / "results"
        out.mkdir(parents=True, exist_ok=True)
        from athena.extraction.llm_extractor import PROMPT_VERSION
        meta = {"split": args.split, "limit": args.limit, "model": llm.model,
                "prompt": PROMPT_VERSION, "chunk_chars": cfg["extraction"]["chunk_chars"],
                "date": time.strftime("%Y-%m-%d")}
        path = out / f"extraction_{args.split}_{len(eligible)}notes.json"
        payload = {
            "meta": meta,
            "results": results,
            "bootstrap": bootstrap_summary,
            "paired_hybrid_vs_rules": paired_summary,
        }
        path.write_text(json.dumps(payload, indent=1))
        print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
