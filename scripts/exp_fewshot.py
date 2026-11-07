"""Experiment: dynamic few-shot prompting (retrieval of similar annotated TRAINING snippets).

    python scripts/exp_fewshot.py val          # tuning split (runs the LLM, ~1 h on the M2)
    python scripts/exp_fewshot.py test --limit 40

For each chunk, the 3 most similar snippets from the n2c2 *training* split (TF-IDF cosine over
word 1-2-grams; runs offline) are inserted as worked examples, with their gold rows in the same
8-column format. The fixed prompt and every other pipeline step are unchanged. Outputs are cached
under a separate prompt version, so the baseline cache is untouched.
Writes docs/results/fewshot_<split>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.eval.n2c2_metrics import Scores, score_note  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.fewshot import COLS, FewShotExtractor  # noqa: E402
from athena.extraction.fewshot import DEFAULT_K as K  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402


def score(notes, llm, mode):
    ex = MedicationExtractor(mode, llm=llm)
    s = Scores()
    for k, n in enumerate(notes):
        score_note(n, ex.extract(n.text).mentions, s)
        if isinstance(llm, FewShotExtractor) and mode == "hybrid" and (k + 1) % 5 == 0:
            print(f"  {k + 1}/{len(notes)} notes, LLM {llm.stats['seconds'] / 60:.0f} min", flush=True)
    return {"strict_f1": round(s.micro("strict").f1, 4), "lenient_f1": round(s.micro("lenient").f1, 4),
            **{t: round(s.strict[t].f1, 4) for t in ("Drug", *COLS)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    cfg = load_config()["extraction"]
    base = OllamaExtractor()
    notes = [n for n in n2c2.load_split(args.split)[: args.limit]
             if all(base._cache_path(c.text).exists() for c in chunk_text(n.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]))]
    fs = FewShotExtractor()
    print(f"{args.split}: {len(notes)} notes; {len(fs.bank)} training snippets in the example bank", flush=True)
    t0 = time.time()
    res = {"notes": len(notes), "k": K}
    res["fewshot_hybrid"] = score(notes, fs, "hybrid")      # runs the LLM (cached afterwards)
    res["fewshot_llm_only"] = score(notes, fs, "llm")
    res["fewshot_llm_seconds"] = round(fs.stats["seconds"])
    res["baseline_hybrid"] = score(notes, base, "hybrid")
    res["baseline_llm_only"] = score(notes, base, "llm")
    res["wall_seconds"] = round(time.time() - t0)
    for k in ("baseline_hybrid", "fewshot_hybrid", "baseline_llm_only", "fewshot_llm_only"):
        print(f"{k:20s} {res[k]}")
    (ROOT / f"docs/results/fewshot_{args.split}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
