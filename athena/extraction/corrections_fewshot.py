"""Level 1 feedback loop: retrieval of similar corrected examples for few-shot prompt augmentation.

Retrieves the 2-3 most similar corrected examples from the correction bank (or training split
for leakage-free offline evaluation) using TF-IDF or rapidfuzz similarity over chunk text, and provides
them as worked examples in the LLM chat prompt. There is no model fine-tuning required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from athena.config import load_config
from athena.data import n2c2
from athena.extraction.llm_extractor import FIELDS
from athena.review.corrections import CorrectionsDB

COLS = ["Strength", "Dosage", "Form", "Route", "Frequency", "Duration"]


def med_dict_to_row(m: dict[str, Any]) -> list[str]:
    """Convert a normalized/corrected medication dict into an 8-column row."""
    drug = str(m.get("name") or m.get("raw") or "").strip()
    attrs = m.get("attributes") or {}
    strength = str(attrs.get("Strength") or m.get("strength") or "")
    dosage = str(attrs.get("Dosage") or m.get("dosage") or "")
    form = str(attrs.get("Form") or m.get("form") or "")
    route = str(attrs.get("Route") or m.get("route") or "")
    frequency = str(attrs.get("Frequency") or m.get("frequency") or "")
    duration = str(attrs.get("Duration") or m.get("duration") or "")
    status = str(m.get("status") or "")
    return [drug, strength, dosage, form, route, frequency, duration, status]


def build_train_corrections_bank(
    split: str = "train",
    lines_per_snippet: int = 8,
    limit: int | None = None,
) -> list[tuple[str, list[list[str]]]]:
    """Build a leakage-free benchmark correction bank strictly from train/dev notes."""
    notes = n2c2.load_split(split)
    if limit:
        notes = notes[:limit]
    bank: list[tuple[str, list[list[str]]]] = []
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
                snippet = note.text[lo:hi].strip()
                if snippet:
                    bank.append((snippet, rows))
    return bank


def build_db_corrections_bank(db_path: Path | str | None = None) -> list[tuple[str, list[list[str]]]]:
    """Build bank from reviewer corrections stored in SQLite (data/corrections.db)."""
    db = CorrectionsDB(db_path)
    entries = db.entries()
    bank: list[tuple[str, list[list[str]]]] = []
    for e in entries:
        text = e.note_text
        if not text:
            # Fall back to reconstructing text snippet from medication strings
            text = "\n".join(str(m.get("raw") or m.get("name") or "") for m in e.corrected_list)
        rows = [med_dict_to_row(m) for m in e.corrected_list]
        if rows and text.strip():
            bank.append((text.strip(), rows))
    return bank


class CorrectionRetriever:
    """TF-IDF or rapidfuzz similarity retriever over a bank of verified/corrected medication examples."""

    def __init__(
        self,
        bank: list[tuple[str, list[list[str]]]] | None = None,
        k: int = 2,
        db_path: Path | str | None = None,
        method: str = "tfidf",
        fallback_to_train: bool = False,
    ):
        self.k = k
        self.method = method
        if bank is not None:
            self.bank = bank
        else:
            db_bank = build_db_corrections_bank(db_path)
            if db_bank:
                self.bank = db_bank
            elif fallback_to_train:
                self.bank = build_train_corrections_bank(split="train")
            else:
                self.bank = []

        if self.bank:
            min_df = 2 if len(self.bank) >= 5 else 1
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform([text for text, _ in self.bank])
            raw_meta = [(t[:100], rows) for t, rows in self.bank]
            self.version = "v_" + hashlib.sha256(json.dumps(raw_meta, sort_keys=True).encode()).hexdigest()[:12]
        else:
            self.vectorizer = None
            self.matrix = None
            self.version = "v0_empty"

    def retrieve(self, text: str, k: int | None = None) -> list[tuple[str, list[list[str]]]]:
        if not self.bank:
            return []
        k = k or self.k
        k = min(k, len(self.bank))

        if self.method == "rapidfuzz":
            try:
                from rapidfuzz import fuzz
                scored = []
                for item in self.bank:
                    score = fuzz.token_set_ratio(text, item[0])
                    if score > 0:
                        scored.append((score, item))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [item for _, item in scored[:k]]
            except ImportError:
                pass  # Fall back to TF-IDF below

        if self.matrix is None or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([text])
        sims = linear_kernel(query_vec, self.matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [self.bank[i] for i in top_indices if sims[i] > 0.0]
