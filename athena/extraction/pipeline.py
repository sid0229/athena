"""Branch A end to end: note text -> grounded medication mentions.

Modes (compared in evaluation):
    llm       LLM rows grounded in text; names propagated to all occurrences
    scispacy  scispaCy CHEMICAL mentions that the drug normaliser recognises
    hybrid    llm + scispaCy drug mentions the LLM missed (default)
If Ollama is unavailable, `hybrid` falls back to `scispacy` and says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from athena.config import load_config
from athena.extraction.align import AlignStats, Mention, Span, ground
from athena.extraction.chunking import chunk_text
from athena.extraction.llm_extractor import OllamaExtractor
from athena.extraction.scispacy_extractor import chemical_spans
from athena.normalize.normalizer import DrugNormalizer, Normalized

CHECKABLE = {"mapped", "ambiguous", "drug_class"}


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
        if mode not in ("llm", "scispacy", "hybrid"):
            raise ValueError(mode)
        self.mode = mode
        self.cfg = load_config()["extraction"]
        self.normalizer = normalizer or DrugNormalizer()
        self.llm = llm if llm is not None else (OllamaExtractor() if mode != "scispacy" else None)

    def _drug_like(self, name: str) -> bool:
        return self.normalizer.normalize(name).status in CHECKABLE

    def extract(self, text: str) -> ExtractionResult:
        t0 = time.perf_counter()
        warnings = []
        mode = self.mode
        if mode != "scispacy" and not self.llm.available():
            warnings.append("Local LLM unavailable — scispaCy-only fallback used.")
            mode = "scispacy"

        mentions: list[Mention] = []
        stats = None
        if mode in ("llm", "hybrid"):
            chunks = chunk_text(text, self.cfg["chunk_chars"], self.cfg["chunk_overlap_chars"])
            rows = [(c, self.llm.extract_chunk(c.text)) for c in chunks]
            mentions, stats = ground(text, rows, self.cfg["fuzzy_ground_threshold"], self._drug_like)

        chem = chemical_spans(text)
        occupied = [(m.drug.start, m.drug.end) for m in mentions]
        agree = set()
        for sp in chem:
            hit = next((o for o in occupied if o[0] < sp.end and sp.start < o[1]), None)
            if hit:
                agree.add(hit)
            elif mode in ("scispacy", "hybrid") and self._drug_like(sp.text):
                mentions.append(Mention(sp, sources={"scispacy"}))
        for m in mentions:
            if (m.drug.start, m.drug.end) in agree:
                m.sources.add("scispacy")
        mentions.sort(key=lambda m: m.drug.start)

        normalized = {(m.drug.start, m.drug.end): self.normalizer.normalize(m.drug.text) for m in mentions}
        return ExtractionResult(text, mentions, mode, stats, time.perf_counter() - t0,
                                normalized, agree, warnings)
