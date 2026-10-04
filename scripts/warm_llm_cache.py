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
    args = ap.parse_args()
    cfg = load_config()["extraction"]
    llm = OllamaExtractor()
    notes = n2c2.load_split(args.split)[: args.limit]
    t0 = time.time()
    for i, note in enumerate(notes, 1):
        for c in chunk_text(note.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]):
            llm.extract_chunk(c.text)
        el = time.time() - t0
        print(f"[{args.split}] {i}/{len(notes)} note {note.id} | {el / 60:.1f} min | "
              f"eta {el / i * (len(notes) - i) / 60:.0f} min | {llm.stats}", flush=True)


if __name__ == "__main__":
    main()
