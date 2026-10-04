"""Deterministic repair of column-shifted LLM rows.

With the compact 8-column output the 3B model sometimes shifts values by one
column on narrative text (e.g. strength="One (1)", form="PO", route="daily").
A value whose surface pattern unambiguously identifies its field is moved to
that field. Ambiguous values stay where the model put them. The repair never
invents text: it only re-labels values the model produced, which are then
grounded in the note as usual.
"""

from __future__ import annotations

import re
from dataclasses import replace

from athena.extraction.llm_extractor import RawMed

_NUM_WORDS = r"(?:one|two|three|four|five|six|half|1/2|\d+(?:[.-]\d+)?)"

PATTERNS: dict[str, re.Pattern] = {
    "strength": re.compile(
        r"^\s*\d[\d.,/\s-]*\s*(?:mg|mcg|g|gm|grams?|units?|u|meq|mmol|ml|%|mg/ml|mcg/hr|mg/kg|iu)\b", re.I),
    "dosage": re.compile(rf"^\s*{_NUM_WORDS}(?:\s*\(\d+(?:[.-]\d+)?\))?\s*(?:tabs?|tablets?|caps?|capsules?|puffs?|drops?|units?)?\s*$", re.I),
    "form": re.compile(
        r"^\s*(?:tab(?:let)?s?|cap(?:sule)?s?|solution|suspension|syrup|patch|cream|ointment|gel|"
        r"inhaler|nebuli[sz]er|spray|drops?|powder|suppositor(?:y|ies)|lozenge|elixir|injection|"
        r"solution for nebulization|disk with device|adhesive patch.*|tablet,?.*|capsule,?.*)\s*$", re.I),
    "route": re.compile(
        r"^\s*(?:po|p\.o\.|iv|i\.v\.|im|sc|sq|subq|subcutaneous(?:ly)?|sl|sublingual|pr|per rectum|"
        r"topical(?:ly)?|inhalation|inhaled|nebulization|neb|intravenous(?:ly)?|intramuscular(?:ly)?|"
        r"oral(?:ly)?|by mouth|nasal|intranasal|ophthalmic|transdermal|gtt|drip|ng|via ng tube|g-tube|pfr)\s*$", re.I),
    "frequency": re.compile(
        r"^\s*(?:q\.?\s?\d+\s?-?\s?\d*\s?h(?:rs?|ours?)?|qd|qod|qhs|hs|bid|tid|qid|daily|nightly|weekly|"
        r"prn|as needed|once|twice|three times|four times|every\b.*|q\s?am|q\s?pm|at bedtime|"
        r"(?:once|twice) (?:a|per) (?:day|week)|\d+ times? (?:a|per) day)\b.*$", re.I),
    "duration": re.compile(r"^\s*(?:for|x|times)\s*\d+.*(?:day|days|week|weeks|month|months|doses?)\s*$", re.I),
}
FIELDS = ("strength", "dosage", "form", "route", "frequency", "duration")


def detect(value: str) -> str | None:
    hits = [f for f, p in PATTERNS.items() if p.search(value)]
    return hits[0] if len(hits) == 1 else None


def repair(row: RawMed) -> tuple[RawMed, int]:
    """Return (repaired row, number of values moved)."""
    values = {f: getattr(row, f) for f in FIELDS}
    out = {f: "" for f in FIELDS}
    leftovers = []
    moved = 0
    for f, v in values.items():
        if not v:
            continue
        kind = detect(v)
        if kind and kind != f and not out[kind]:
            out[kind] = v
            moved += 1
        else:
            leftovers.append((f, v))
    for f, v in leftovers:
        if not out[f]:
            out[f] = v
    return replace(row, **out), moved
