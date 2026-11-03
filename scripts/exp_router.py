"""Experiment: dynamic router — send only narrative chunks to the LLM, list-like chunks to rules.

    python scripts/exp_router.py

Each LLM chunk gets a "list score" = share of its non-empty lines that look like medication-list
or structured lines (numbered/bulleted items, dose units, 'Sig:', short key: value lines).
Chunks with score >= threshold skip the LLM (rules + dictionary + scispaCy still run on the
whole note). Uses cached LLM outputs only, so every setting is scored on identical notes.
The threshold is chosen on the validation split and then applied once to the test subset.
Writes docs/results/router_experiment.json.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.eval.n2c2_metrics import Scores, score_note  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import OllamaExtractor  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402

LISTY = re.compile(r"^\s*(\d{1,2}[.)]|[-*•])\s|\b\d+(\.\d+)?\s?(mg|mcg|g|units?|ml|meq)\b|\bsig:|\b(q\d+h|bid|tid|qid|qhs|prn|daily)\b"
                   r"|^[A-Za-z ]{2,30}:\s*\S", re.I)
THRESHOLDS = [1.01, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]   # 1.01 = never skip (current hybrid), 0.0 = always skip


def list_score(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    return sum(bool(LISTY.search(l)) for l in lines) / len(lines) if lines else 1.0


class RoutedLLM(OllamaExtractor):
    def __init__(self, thr: float):
        super().__init__()
        self.thr, self.sent, self.skipped, self.chars_sent, self.chars_total = thr, 0, 0, 0, 0

    def available(self) -> bool:  # cached outputs only; no server needed
        return True

    def extract_chunk(self, text):
        self.chars_total += len(text)
        if list_score(text) >= self.thr:
            self.skipped += 1
            return []
        self.sent += 1
        self.chars_sent += len(text)
        return super().extract_chunk(text)


def run(notes, thr):
    llm = RoutedLLM(thr)
    ex = MedicationExtractor("hybrid", llm=llm)
    s = Scores()
    for n in notes:
        score_note(n, ex.extract(n.text).mentions, s)
    tot = llm.sent + llm.skipped
    return {"threshold": thr, "strict_f1": round(s.micro("strict").f1, 4), "lenient_f1": round(s.micro("lenient").f1, 4),
            "drug_f1": round(s.strict["Drug"].f1, 4), "chunks_to_llm": llm.sent, "chunks_total": tot,
            "llm_share_chars": round(llm.chars_sent / llm.chars_total, 3) if llm.chars_total else 0}


def cached(split, limit):
    cfg = load_config()["extraction"]
    llm = OllamaExtractor()
    return [n for n in n2c2.load_split(split)[:limit]
            if all(llm._cache_path(c.text).exists() for c in chunk_text(n.text, cfg["chunk_chars"], cfg["chunk_overlap_chars"]))]


def main():
    val, test = cached("val", None), cached("test", 40)
    print(f"val {len(val)} notes, test {len(test)} notes (cached)")
    out = {"val": [], "test": []}
    for thr in THRESHOLDS:
        r = run(val, thr)
        out["val"].append(r)
        print("val ", r)
    full = out["val"][0]["strict_f1"]
    # pick the threshold that sends the fewest chunks to the LLM while losing <= 0.005 strict F1 on val
    ok = [r for r in out["val"] if r["strict_f1"] >= full - 0.005]
    best = min(ok, key=lambda r: r["llm_share_chars"])
    out["chosen_threshold"] = best["threshold"]
    for thr in sorted({1.01, best["threshold"], 0.0}, reverse=True):
        r = run(test, thr)
        out["test"].append(r)
        print("test", r)
    (ROOT / "docs/results/router_experiment.json").write_text(json.dumps(out, indent=1))
    print("chosen threshold", best["threshold"])


if __name__ == "__main__":
    main()
