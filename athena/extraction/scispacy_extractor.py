"""scispaCy drug-mention detector: fallback when the LLM is unavailable, and
cross-check signal for extraction confidence (README §2.1)."""

from __future__ import annotations

from functools import lru_cache

import spacy

from athena.config import load_config
from athena.extraction.align import Span


@lru_cache(maxsize=1)
def _nlp():
    # Only the NER component is needed.
    return spacy.load(load_config()["extraction"]["scispacy_model"], disable=["parser", "lemmatizer"])


def chemical_spans(text: str) -> list[Span]:
    nlp = _nlp()
    nlp.max_length = max(nlp.max_length, len(text) + 1)
    return [Span(e.start_char, e.end_char, e.text) for e in nlp(text).ents if e.label_ == "CHEMICAL"]
