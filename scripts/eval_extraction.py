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
    args = ap.parse_args()

    cfg = load_config()
    notes = n2c2.load_split(args.split)[: args.limit]
    llm = OllamaExtractor()
    if args.run_llm:
        eligible = list(notes)
    else:
        eligible = [n for n in notes if fully_cached(llm, n.text, cfg["extraction"])]
    print(f"{args.split}: {len(notes)} notes, {len(eligible)} with cached LLM output")

    results = {}
    for mode in args.modes.split(","):
        ex = MedicationExtractor(mode, llm=llm)
        use = notes if mode == "scispacy" else eligible
        scores, t0, dropped, total = Scores(), time.time(), 0, 0
        for note in use:
            r = ex.extract(note.text)
            score_note(note, r.mentions, scores)
            if r.stats:
                dropped += len(r.stats.values_dropped)
                total += r.stats.values_total
        print(table(f"{mode} ({len(use)} notes, {time.time() - t0:.0f}s)", scores))
        if total:
            print(f"LLM attribute values not found in text (dropped): {dropped}/{total} ({dropped / total:.1%})")
        results[mode] = {"notes": [n.id for n in use], "n_notes": len(use), **scores.as_dict(),
                         "llm_values_dropped": dropped, "llm_values_total": total}

    if args.save:
        out = ROOT / "docs" / "results"
        out.mkdir(parents=True, exist_ok=True)
        meta = {"split": args.split, "limit": args.limit, "model": llm.model,
                "prompt": "extract_v1", "chunk_chars": cfg["extraction"]["chunk_chars"],
                "date": time.strftime("%Y-%m-%d")}
        path = out / f"extraction_{args.split}{'_' + str(args.limit) if args.limit else ''}.json"
        path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
