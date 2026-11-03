"""Dynamic few-shot LLM extractor using TF-IDF retrieval over training notes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from athena.data import n2c2
from athena.extraction.llm_extractor import _SCHEMA, OllamaExtractor, _salvage_rows

VERSION = "extract_v1_fewshot3"
DEFAULT_K = 3
DEFAULT_LINES = 8
COLS = ["Strength", "Dosage", "Form", "Route", "Frequency", "Duration"]


def build_snippet_bank(lines_per_snippet: int = DEFAULT_LINES) -> list[tuple[str, list[list[str]]]]:
    """Extract line windows from training notes containing at least one gold drug with attributes."""
    notes = n2c2.load_split("train")
    bank = []
    for note in notes:
        lines = note.text.split("\n")
        pos = 0
        offsets = []
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1

        for i in range(0, len(lines), lines_per_snippet):
            lo = offsets[i]
            hi = (
                offsets[min(i + lines_per_snippet, len(lines) - 1)]
                if i + lines_per_snippet < len(lines)
                else len(note.text)
            )
            rows = []
            for drug in note.drugs():
                if lo <= drug.start and drug.end <= hi:
                    attrs = {
                        a.type: a.text
                        for a in note.attributes_of(drug.id)
                        if lo <= a.start and a.end <= hi
                    }
                    rows.append([drug.text] + [attrs.get(c, "") for c in COLS] + [""])
            if rows:
                bank.append((note.text[lo:hi].strip(), rows))
    if not bank:
        raise ValueError(
            "Few-shot example bank is empty. Ensure the n2c2 'train' split is present and annotated."
        )
    return bank


class FewShotExtractor(OllamaExtractor):
    """Ollama extractor augmented with dynamically retrieved training examples."""

    def __init__(
        self,
        k: int = DEFAULT_K,
        lines_per_snippet: int = DEFAULT_LINES,
        model: str | None = None,
        use_cache: bool = True,
        bank: list[tuple[str, list[list[str]]]] | None = None,
    ):
        super().__init__(model=model, use_cache=use_cache)
        self.k = k
        self.cache_dir = self.cache_dir.parent / f"{self.model.replace(':', '_')}_{VERSION}"
        self.bank = build_snippet_bank(lines_per_snippet=lines_per_snippet) if bank is None else bank
        if not self.bank:
            raise ValueError(
                "Few-shot example bank is empty. Cannot initialize FewShotExtractor without examples."
            )
        min_df = 2 if len(self.bank) >= 5 else 1
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([text for text, _ in self.bank])

    def _cache_path(self, text: str) -> Path:
        token = hashlib.sha256(f"{self.model}|{VERSION}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{token}.json"

    def retrieve_examples(self, text: str) -> list[tuple[str, list[list[str]]]]:
        sims = linear_kernel(self.vectorizer.transform([text]), self.matrix)[0]
        top_indices = sims.argsort()[::-1][: self.k]
        return [self.bank[i] for i in top_indices]

    def examples(self, text: str) -> list[tuple[str, list[list[str]]]]:
        """Alias for retrieve_examples."""
        return self.retrieve_examples(text)

    def _call(self, text: str) -> list[list[str]] | None:
        messages = [{"role": "system", "content": self.system}]
        for ex_text, rows in self.retrieve_examples(text):
            messages.extend([
                {"role": "user", "content": ex_text},
                {"role": "assistant", "content": json.dumps({"meds": rows})},
            ])
        messages.append({"role": "user", "content": text})

        for attempt in range(2):
            seed = self.options["seed"] if attempt == 0 else self.options["seed"] + 1
            options = {**self.options, "seed": seed}
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=_SCHEMA,
                options=options,
            )
            self.stats["calls"] += 1
            self.stats["seconds"] += (response.total_duration or 0) / 1e9
            self.stats["gen_tokens"] = self.stats.get("gen_tokens", 0) + (response.eval_count or 0)

            try:
                return json.loads(response.message.content)["meds"]
            except (json.JSONDecodeError, KeyError, TypeError):
                self.stats["parse_failures"] = self.stats.get("parse_failures", 0) + 1
                salvaged = _salvage_rows(response.message.content)
                if salvaged:
                    self.stats["salvaged"] = self.stats.get("salvaged", 0) + 1
                    return salvaged
        return None
