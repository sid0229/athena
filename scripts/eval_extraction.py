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


def _f1(rows) -> float:
    tp, fp, fn = (sum(x[i] for x in rows) for i in range(3))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def bootstrap_f1(rows, n=1000, seed=42) -> list[float]:
    """95% CI of micro F1, resampling notes (the unit that was sampled)."""
    import random
    rng = random.Random(seed)
    vals = sorted(_f1([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n))
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def paired_delta(a, b, n=1000, seed=42) -> dict:
    """Difference in micro F1 between two systems scored on the same notes, paired bootstrap."""
    import random
    rng = random.Random(seed)
    idx = range(len(a))
    vals = sorted(_f1([a[i] for i in s]) - _f1([b[i] for i in s])
                  for s in ([rng.randrange(len(a)) for _ in idx] for _ in range(n)))
    return {"delta": round(_f1(a) - _f1(b), 4), "ci95": [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)],
            "share_of_resamples_above_0": round(sum(v > 0 for v in vals) / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--modes", default="scispacy,rules,llm,hybrid")
    ap.add_argument("--run-llm", action="store_true")
    ap.add_argument("--save", action="store_true")
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

    results = {}
    for mode in args.modes.split(","):
        ex = MedicationExtractor(mode, llm=llm)
        use = eligible  # every mode on the same notes, so scores are comparable
        scores, t0, dropped, total = Scores(), time.time(), 0, 0
        per_note = []  # (tp, fp, fn) strict micro per note, for the bootstrap CI
        for note in use:
            r = ex.extract(note.text)
            assert r.mode == mode, f"{note.id}: scored as {r.mode}, not {mode} ({r.warnings})"
            score_note(note, r.mentions, scores)
            one = score_note(note, r.mentions).micro("strict")
            per_note.append((one.tp, one.fp, one.fn))
            if r.stats:
                dropped += len(r.stats.values_dropped)
                total += r.stats.values_total
        print(table(f"{mode} ({len(use)} notes, {time.time() - t0:.0f}s)", scores))
        if total:
            print(f"LLM attribute values not found in text (dropped): {dropped}/{total} ({dropped / total:.1%})")
        ci = bootstrap_f1(per_note)
        print(f"strict micro F1 95% CI (bootstrap over notes): {ci}")
        results[mode] = {"notes": [n.id for n in use], "n_notes": len(use), **scores.as_dict(),
                         "strict_micro_f1_ci95": ci, "per_note_strict": per_note,
                         "llm_values_dropped": dropped, "llm_values_total": total}
    if {"hybrid", "rules"} <= results.keys():
        d = paired_delta(results["hybrid"]["per_note_strict"], results["rules"]["per_note_strict"])
        print(f"hybrid − rules strict F1: {d['delta']:+.4f}, 95% CI {d['ci95']} (paired bootstrap over notes)")
        results["hybrid_minus_rules"] = d

    if args.save:
        out = ROOT / "docs" / "results"
        out.mkdir(parents=True, exist_ok=True)
        from athena.extraction.llm_extractor import PROMPT_VERSION
        meta = {"split": args.split, "limit": args.limit, "model": llm.model,
                "prompt": PROMPT_VERSION, "chunk_chars": cfg["extraction"]["chunk_chars"],
                "date": time.strftime("%Y-%m-%d")}
        path = out / f"extraction_{args.split}_{len(eligible)}notes.json"
        path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
