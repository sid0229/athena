"""Run the LLM over n2c2 notes to fill the on-disk cache (no scoring).

    python scripts/warm_llm_cache.py val
    python scripts/warm_llm_cache.py test --limit 40
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=None, help="override LLM model name")
    ap.add_argument("--use-corrections", action="store_true")
    ap.add_argument("--corrections-split", default="train", choices=["train", "val"])
    args = ap.parse_args()
    cfg = load_config()["extraction"]
    if args.use_corrections:
        from athena.extraction.corrections_fewshot import CorrectionRetriever, build_train_corrections_bank
        bank = build_train_corrections_bank(split=args.corrections_split)
        retriever = CorrectionRetriever(bank=bank, k=2)
        llm = OllamaExtractor(model=args.model, use_corrections=True, retriever=retriever)
    else:
        llm = OllamaExtractor(model=args.model, use_corrections=False)
    notes = n2c2.load_split(args.split)[: args.limit]
    t0 = time.time()
    for i, note in enumerate(notes, 1):
        for c in chunk_text(note.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]):
            try:
                llm.extract_chunk(c.text)
            except RuntimeError:
                raise  # stale cache signature
            except Exception as e:  # one timeout must not end a multi-hour run
                print(f"[WARN] note {note.id} chunk failed ({type(e).__name__}: {e}); continuing...", file=sys.stderr, flush=True)
        el = time.time() - t0
        print(f"[{args.split}] {i}/{len(notes)} note {note.id} | {el / 60:.1f} min | "
              f"eta {el / i * (len(notes) - i) / 60:.0f} min | {llm.stats}", flush=True)


if __name__ == "__main__":
    main()
