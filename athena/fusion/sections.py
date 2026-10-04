"""Locate medication-list sections in a discharge summary."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER = re.compile(r"^[ \t]*([A-Za-z][A-Za-z /&'-]{2,45}):", re.M)

ADMISSION = re.compile(
    r"^(medications on admission|admission medications|meds on admission|home medications|home meds|"
    r"meds at home|medications at home|outpatient medications|pre-?admission medications|meds on transfer|"
    r"transfer medications)$", re.I)
DISCHARGE = re.compile(
    r"^(discharge medications|discharge meds|medications on discharge|meds on discharge)$", re.I)
# "Headers" that appear *inside* a medication list and must not end it, e.g.
# "Nebulization Sig:" or "Disp:" — matched on the last word of the title.
_INLINE = re.compile(r"(?:^|\s)(sig|disp|refills?|qty|dose|instructions)$", re.I)


@dataclass(frozen=True)
class Section:
    kind: str   # "admission" | "discharge"
    title: str
    start: int  # start of the section body
    end: int


def medication_sections(text: str) -> list[Section]:
    headers = [(m.start(), m.end(), m.group(1).strip()) for m in _HEADER.finditer(text)
               if not _INLINE.search(m.group(1).strip())]
    out = []
    for i, (hs, he, title) in enumerate(headers):
        kind = "admission" if ADMISSION.match(title) else "discharge" if DISCHARGE.match(title) else None
        if kind:
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            out.append(Section(kind, title, he, end))
    return out


def section_of(pos: int, sections: list[Section]) -> Section | None:
    return next((s for s in sections if s.start <= pos < s.end), None)
