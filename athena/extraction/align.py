"""Ground LLM output in the note text (anti-hallucination guard, README §2.1).

1. Drug names returned by the LLM are located in their own chunk (exact,
   case-insensitive, whitespace-flexible). Names never found are dropped.
2. Grounded names that look like real drugs are then marked at *every*
   occurrence in the note (n2c2 annotates every mention).
3. For each LLM row, the occurrence with the most of its attribute values nearby
   becomes the anchor; attribute values are searched only inside that drug's own
   window (up to the next drug / list item). Values not found are dropped.
4. n2c2 convention: a frequency span extends over a following "( ... )"
   explanation and "as needed".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from athena.extraction.chunking import Chunk
from athena.extraction.llm_extractor import RawMed
from athena.extraction.slot_repair import repair

ATTR_TYPES = {
    "strength": "Strength", "dosage": "Dosage", "form": "Form",
    "route": "Route", "frequency": "Frequency", "duration": "Duration",
}
# Precedence when attribute spans would overlap.
ATTR_ORDER = ("strength", "form", "route", "frequency", "dosage", "duration")

# Strings an LLM sometimes returns as a "drug" that must never be propagated.
NOT_DRUGS = {
    "tablet", "tablets", "capsule", "capsules", "sig", "po", "iv", "daily", "bid", "solution",
    "medications", "medication", "meds", "discharge", "home", "none", "unknown", "allergies",
    "patch", "injection", "drip", "gtt", "inhalation", "dose", "doses", "mg", "the patient",
}

WINDOW_BEFORE = 40
WINDOW_AFTER = 300
_ITEM_BOUNDARY = re.compile(r"\n\s*\d{1,2}[.)]\s")
_FREQ_TAIL = re.compile(r"\s*\([^()\n]{1,40}\)(?:\s*\.?\s*as needed)?|\s+as needed", re.I)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str


@dataclass
class Mention:
    drug: Span
    attributes: dict[str, list[Span]] = field(default_factory=dict)  # n2c2 type -> spans
    status: str = ""
    sources: set[str] = field(default_factory=set)  # llm | propagated | scispacy
    llm_row: RawMed | None = None


@dataclass
class AlignStats:
    rows: int = 0
    drugs_not_found: int = 0
    values_total: int = 0
    values_grounded: int = 0
    values_fuzzy: int = 0
    values_moved: int = 0  # column-shift repairs (slot_repair)
    values_dropped: list[tuple[str, str, str]] = field(default_factory=list)  # (drug, field, value)


def _pattern(value: str) -> re.Pattern:
    parts = [re.escape(p) for p in value.split()]
    body = r"\s+".join(parts)
    pre = r"(?<![A-Za-z0-9])" if value[:1].isalnum() else ""
    post = r"(?![A-Za-z0-9])" if value[-1:].isalnum() else ""
    return re.compile(pre + body + post, re.I)


def _find_all(text: str, value: str, lo: int = 0, hi: int | None = None) -> list[tuple[int, int]]:
    hi = len(text) if hi is None else hi
    return [(m.start(), m.end()) for m in _pattern(value).finditer(text, lo, hi)]


def _fuzzy_find(text: str, value: str, lo: int, hi: int, threshold: float) -> tuple[int, int] | None:
    if len(value) < 4 or hi <= lo:
        return None
    window = text[lo:hi]
    al = fuzz.partial_ratio_alignment(value.lower(), window.lower(), score_cutoff=threshold)
    if al is None:
        return None
    return lo + al.dest_start, lo + al.dest_end


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def ground(
    text: str,
    chunk_rows: list[tuple[Chunk, list[RawMed]]],
    fuzzy_threshold: float = 90,
    is_drug_like=None,
) -> tuple[list[Mention], AlignStats]:
    """Return one Mention per drug occurrence in `text`.

    `is_drug_like(name) -> bool` gates propagation of a grounded LLM drug name to
    its other occurrences (e.g. normaliser-mapped or scispaCy-confirmed names).
    """
    stats = AlignStats()

    # 1. Ground drug names inside their own chunk.
    grounded_rows: list[tuple[Chunk, RawMed]] = []
    names: set[str] = set()
    for chunk, rows in chunk_rows:
        for row in rows:
            stats.rows += 1
            row, moved = repair(row)
            stats.values_moved += moved
            name = row.drug.strip(" .,;:-")
            if len(name) < 2 or name.lower() in NOT_DRUGS:
                stats.drugs_not_found += 1
                continue
            if not _find_all(text, name, chunk.start, chunk.end):
                stats.drugs_not_found += 1
                continue
            grounded_rows.append((chunk, row))
            names.add(name.lower())

    # 2. Every occurrence of every grounded name; longest match wins on overlap.
    occ: list[tuple[int, int]] = []
    for name in sorted(names, key=len, reverse=True):
        propagate = is_drug_like(name) if is_drug_like else True
        for s, e in _find_all(text, name):
            if any(_overlaps((s, e), o) for o in occ):
                continue
            occ.append((s, e))
        if not propagate:
            # keep only occurrences inside chunks where the LLM actually listed it
            chunks = [c for c, r in grounded_rows if r.drug.strip(" .,;:-").lower() == name]
            occ = [o for o in occ
                   if text[o[0]:o[1]].lower() != name or any(c.start <= o[0] < c.end for c in chunks)]
    occ.sort()
    mentions = {o: Mention(Span(o[0], o[1], text[o[0]:o[1]]), sources={"propagated"}) for o in occ}
    starts = [o[0] for o in occ]

    def window(o: tuple[int, int]) -> tuple[int, int]:
        i = starts.index(o[0])
        lo = max(o[0] - WINDOW_BEFORE, occ[i - 1][1] if i > 0 else 0)
        hi = min(o[1] + WINDOW_AFTER, occ[i + 1][0] if i + 1 < len(occ) else len(text))
        m = _ITEM_BOUNDARY.search(text, o[1], hi)
        return lo, (m.start() if m else hi)

    # 3. Attach attributes of each LLM row to its best occurrence.
    taken: list[tuple[int, int]] = list(occ)
    for chunk, row in grounded_rows:
        name = row.drug.strip(" .,;:-").lower()
        cands = [o for o in occ if text[o[0]:o[1]].lower() == name and chunk.start <= o[0] < chunk.end]
        if not cands:
            continue
        attrs = row.attributes()

        def score(o):
            lo, hi = window(o)
            free = mentions[o].llm_row is None
            return (sum(bool(_find_all(text, v, lo, hi)) for v in attrs.values()), free, -o[0])

        best = max(cands, key=score)
        m = mentions[best]
        m.sources.add("llm")
        m.sources.discard("propagated")
        if m.llm_row is not None:
            continue  # occurrence already anchored by another row
        m.llm_row = row
        m.status = row.status.lower()
        lo, hi = window(best)
        for f in ATTR_ORDER:
            v = attrs.get(f)
            if not v:
                continue
            stats.values_total += 1
            spans = [sp for sp in _find_all(text, v, lo, hi) if not any(_overlaps(sp, t) for t in taken)]
            if not spans:
                fz = _fuzzy_find(text, v, lo, hi, fuzzy_threshold)
                if fz and not any(_overlaps(fz, t) for t in taken):
                    spans = [fz]
                    stats.values_fuzzy += 1
            if not spans:
                stats.values_dropped.append((row.drug, f, v))
                continue
            stats.values_grounded += 1
            out = []
            for s, e in spans:
                if f == "frequency" and (t := _FREQ_TAIL.match(text, e)) and t.end() <= hi:
                    e = t.end()
                out.append(Span(s, e, text[s:e]))
                taken.append((s, e))
            m.attributes.setdefault(ATTR_TYPES[f], []).extend(out)

    return [mentions[o] for o in occ], stats
