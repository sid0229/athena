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
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import linear_kernel  # noqa: E402

from athena.config import ROOT, load_config  # noqa: E402
from athena.data import n2c2  # noqa: E402
from athena.eval.n2c2_metrics import Scores, score_note  # noqa: E402
from athena.extraction.chunking import chunk_text  # noqa: E402
from athena.extraction.llm_extractor import _SCHEMA, OllamaExtractor, _salvage_rows  # noqa: E402
from athena.extraction.pipeline import MedicationExtractor  # noqa: E402

VERSION, K, LINES = "extract_v1_fewshot3", 3, 8
COLS = ["Strength", "Dosage", "Form", "Route", "Frequency", "Duration"]


def snippets():
    """8-line windows from training notes that contain at least one gold drug, with gold rows."""
    out = []
    for note in n2c2.load_split("train"):
        lines, pos = note.text.split("\n"), 0
        offs = []
        for ln in lines:
            offs.append(pos)
            pos += len(ln) + 1
        for i in range(0, len(lines), LINES):
            lo = offs[i]
            hi = offs[min(i + LINES, len(lines) - 1)] if i + LINES < len(lines) else len(note.text)
            rows = []
            for d in note.drugs():
                if lo <= d.start and d.end <= hi:
                    attrs = {a.type: a.text for a in note.attributes_of(d.id) if lo <= a.start and a.end <= hi}
                    rows.append([d.text] + [attrs.get(c, "") for c in COLS] + [""])
            if rows:
                out.append((note.text[lo:hi].strip(), rows))
    return out


class FewShotExtractor(OllamaExtractor):
    def __init__(self):
        super().__init__()
        self.cache_dir = self.cache_dir.parent / f"{self.model.replace(':', '_')}_{VERSION}"
        self.bank = snippets()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.mat = self.vec.fit_transform([t for t, _ in self.bank])

    def _cache_path(self, text):
        return self.cache_dir / (hashlib.sha256(f"{self.model}|{VERSION}|{text}".encode()).hexdigest()[:24] + ".json")

    def examples(self, text):
        sims = linear_kernel(self.vec.transform([text]), self.mat)[0]
        return [self.bank[i] for i in sims.argsort()[::-1][:K]]

    def _call(self, text):
        msgs = [{"role": "system", "content": self.system}]
        for ex_text, rows in self.examples(text):
            msgs += [{"role": "user", "content": ex_text},
                     {"role": "assistant", "content": json.dumps({"meds": rows})}]
        msgs.append({"role": "user", "content": text})
        for attempt in range(2):
            resp = self.client.chat(model=self.model, messages=msgs, format=_SCHEMA,
                                    options=self.options if attempt == 0 else {**self.options, "seed": self.options["seed"] + 1})
            self.stats["calls"] += 1
            self.stats["seconds"] += (resp.total_duration or 0) / 1e9
            try:
                return json.loads(resp.message.content)["meds"]
            except (json.JSONDecodeError, KeyError, TypeError):
                rows = _salvage_rows(resp.message.content)
                if rows:
                    return rows
        return None


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
