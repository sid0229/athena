"""Branch A end to end: note text -> grounded medication mentions.

Modes (compared in evaluation):
    scispacy  scispaCy CHEMICAL mentions the normaliser recognises (baseline, no attributes)
    llm       LLM rows grounded in text; names propagated to all occurrences
    rules     no LLM: scispaCy + drug dictionary + rule-based attributes
    hybrid    llm + rules (default): LLM mentions/attributes first, then dictionary /
              scispaCy mentions it missed, then rule attributes for slots still empty
If Ollama is unavailable, `hybrid`/`llm` fall back to `rules` and say so.
Detections inside lab/results sections are skipped for the non-LLM detectors.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from athena.config import load_config
from athena.extraction.align import WINDOW_AFTER, AlignStats, Mention, Span, ground
from athena.extraction.chunking import chunk_text
from athena.extraction.llm_extractor import OllamaExtractor
from athena.extraction.rules import (AMBIGUOUS_WORDS, dictionary_spans, in_ranges, lab_sections,
                                     rule_attributes)
from athena.extraction.scispacy_extractor import chemical_spans
from athena.normalize.normalizer import DrugNormalizer, Normalized

CHECKABLE = {"mapped", "ambiguous", "drug_class"}
MODES = ("scispacy", "llm", "rules", "hybrid")
RULE_WINDOW_BEFORE = 15
# Attribute types where rule spans take precedence over LLM spans in hybrid mode
# (chosen on the n2c2 *validation* split: rules had higher F1 for these types).
RULE_FIRST = {"Dosage", "Route", "Duration", "Form"}
# Types where LLM values are dropped entirely in hybrid mode (LLM fallback values
# were mostly wrong on the validation split).
RULE_ONLY = {"Dosage", "Duration"}
_ITEM_BOUNDARY = re.compile(r"\n\s*\d{1,2}[.)]\s|\n\s*\n")


@dataclass
class ExtractionResult:
    text: str
    mentions: list[Mention]
    mode: str
    stats: AlignStats | None = None
    seconds: float = 0.0
    normalized: dict[tuple[int, int], Normalized] = field(default_factory=dict)
    scispacy_agree: set[tuple[int, int]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


class MedicationExtractor:
    def __init__(self, mode: str = "hybrid", llm: OllamaExtractor | None = None,
                 normalizer: DrugNormalizer | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.cfg = load_config()["extraction"]
        self.normalizer = normalizer or DrugNormalizer()
        uses_llm = mode in ("llm", "hybrid")
        self.llm = llm if llm is not None else (OllamaExtractor() if uses_llm else None)

    def _drug_like(self, name: str) -> bool:
        return self.normalizer.normalize(name).status in CHECKABLE

    def extract(self, text: str) -> ExtractionResult:
        t0 = time.perf_counter()
        warnings = []
        mode = self.mode
        if mode in ("llm", "hybrid") and not self.llm.available():
            warnings.append("Local LLM unavailable — rule-based fallback (scispaCy + dictionary + rules) used.")
            mode = "rules"

        mentions: list[Mention] = []
        stats = None
        if mode in ("llm", "hybrid"):
            chunks = chunk_text(text, self.cfg["chunk_chars"], self.cfg["chunk_overlap_chars"])
            rows = [(c, self.llm.extract_chunk(c.text)) for c in chunks]
            mentions, stats = ground(text, rows, self.cfg["fuzzy_ground_threshold"], self._drug_like)

        labs = lab_sections(text) if mode != "llm" else []
        occupied = [(m.drug.start, m.drug.end) for m in mentions]

        def add(sp: Span, source: str) -> None:
            if any(sp.start < b and a < sp.end for a, b in occupied):
                return
            mentions.append(Mention(sp, sources={source}))
            occupied.append((sp.start, sp.end))

        # scispaCy: agreement signal always; new mentions in non-llm modes.
        chem = chemical_spans(text)
        agree = set()
        for sp in chem:
            hit = next((o for o in occupied if o[0] < sp.end and sp.start < o[1]), None)
            if hit:
                agree.add(hit)
            elif (mode != "llm" and not in_ranges(sp.start, labs)
                  and sp.text.lower() not in AMBIGUOUS_WORDS and self._drug_like(sp.text)):
                add(sp, "scispacy")
        if mode in ("rules", "hybrid"):
            for sp in dictionary_spans(text, skip=labs):
                add(sp, "dictionary")
        for m in mentions:
            if (m.drug.start, m.drug.end) in agree:
                m.sources.add("scispacy")
        mentions.sort(key=lambda m: m.drug.start)

        if mode == "hybrid":
            for m in mentions:
                for t in RULE_ONLY:
                    m.attributes.pop(t, None)
        if mode in ("rules", "hybrid"):
            self._rule_attributes(text, mentions, labs, RULE_FIRST if mode == "hybrid" else set())

        normalized = {(m.drug.start, m.drug.end): self.normalizer.normalize(m.drug.text) for m in mentions}
        return ExtractionResult(text, mentions, mode, stats, time.perf_counter() - t0,
                                normalized, agree, warnings)

    @staticmethod
    def _rule_attributes(text: str, mentions: list[Mention], labs, rule_first: set[str]) -> None:
        """Rule attributes in each drug's own window. For `rule_first` types, rule spans
        replace LLM spans when rules find something; otherwise rules only fill empty types."""
        taken = [(m.drug.start, m.drug.end) for m in mentions]
        taken += [(s.start, s.end) for m in mentions for t, spans in m.attributes.items()
                  if t not in rule_first for s in spans]
        for i, m in enumerate(mentions):
            if in_ranges(m.drug.start, labs):
                continue
            prev_end = mentions[i - 1].drug.end if i else 0
            next_start = mentions[i + 1].drug.start if i + 1 < len(mentions) else len(text)
            lo = max(m.drug.start - RULE_WINDOW_BEFORE, prev_end)
            hi = min(m.drug.end + WINDOW_AFTER, next_start)
            b = _ITEM_BOUNDARY.search(text, m.drug.end, hi)
            hi = b.start() if b else hi
            found = rule_attributes(text, lo, hi, taken)
            for t, spans in found.items():
                if t not in m.attributes or t in rule_first:
                    m.attributes[t] = spans
                    m.sources.add("rules")
            # LLM spans kept for rule-first types now occupy their positions.
            taken += [(s.start, s.end) for t, spans in m.attributes.items() if t in rule_first for s in spans]
