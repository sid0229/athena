"""Locate medication-list sections in a discharge summary.

A medication section starts at a known header ("Discharge Medications:", "Medications on
Admission:", also singular, with a parenthetical, or as an all-caps line without colon) and ends at
the next *real* section header, or at a paragraph of prose after a blank line. Lines inside the list
that merely contain a colon ("Insulin Lispro: sliding scale", "Note: hold if HR < 60", "Sig:") do not
end it — they used to, which dropped the rest of the list (code review, Nov 2026).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "Title:" at the start of a line, or an ALL-CAPS line on its own ("DISCHARGE MEDICATIONS").
_HEADER = re.compile(r"^[ \t]*(?:(?P<t>[A-Za-z][A-Za-z0-9 /&'(),-]{2,60}?)[ \t]*:|(?P<caps>[A-Z][A-Z /&'-]{3,50})[ \t]*$)",
                     re.M)
_PAREN = re.compile(r"\s*\([^)]*\)\s*")

ADMISSION = re.compile(
    r"^(medications? on admission|admission medications?|meds on admission|admission meds|home medications?|"
    r"home meds|meds at home|medications? at home|outpatient medications?|outpatient meds|"
    r"pre-?admission medications?|medications? prior to admission|prior to admission medications?|"
    r"meds prior to admission|meds on transfer|medications? on transfer|transfer medications?|transfer meds)$",
    re.I)
DISCHARGE = re.compile(
    r"^(discharge medications?|discharge meds|medications? on discharge|meds on discharge|"
    r"medications? at discharge|discharge medication list)$", re.I)
# Titles that occur *inside* a medication list and never end it (last word of the title).
_INLINE = re.compile(r"(?:^|\s)(sig|disp|refills?|qty|dose|doses|instructions|note|notes|hold|take|give|"
                     r"start|stop|indication|duration|route|frequency|tablets?|capsules?|patch|scale)$", re.I)
# Section titles that always end a medication list.
_KNOWN_END = re.compile(
    r"^(discharge (disposition|diagnos[ie]s|condition|instructions|followup|follow-up)|disposition|"
    r"facility|follow ?-?up( instructions)?|followup instructions|attending|physical exam|pertinent results|"
    r"brief hospital course|hospital course|allergies|chief complaint|history|past medical history|"
    r"past surgical history|social history|family history|major surgical or invasive procedure|"
    r"labs?|imaging|studies|assessment|plan|assessment and plan|impression|condition at discharge|"
    r"diagnos[ie]s|instructions|patient instructions|signed|dictated by|completed by|service|procedures?|"
    r"wound care|diet|activity|code status|primary diagnosis|secondary diagnos[ie]s)$", re.I)
_DOSEY = re.compile(r"\d|\b(mg|mcg|units?|tabs?|caps?|po|iv|daily|bid|tid|prn|sig)\b", re.I)


@dataclass(frozen=True)
class Section:
    kind: str   # "admission" | "discharge"
    title: str
    start: int  # start of the section body
    end: int


def _title(m: re.Match) -> str:
    return _PAREN.sub(" ", m.group("t") or m.group("caps")).strip()


def _kind(title: str) -> str | None:
    return "admission" if ADMISSION.match(title) else "discharge" if DISCHARGE.match(title) else None


_ITEM = re.compile(r"^[ \t]*(\d+[.)]|[-*•])\s|\bsig\s*:", re.I | re.M)


def _prose_break(text: str, start: int, end: int) -> int:
    """End of the list at a blank line followed by a prose sentence (≥ 5 words, no doses), unless
    more list items (numbered lines, "Sig:", dose lines) follow it before the section's end."""
    body = text[start:end]
    for m in re.finditer(r"\n[ \t]*\n[ \t]*(?P<line>[^\n]+)", body):
        line = m.group("line").strip()
        if len(line.split()) >= 5 and not _ITEM.search(line) and not _DOSEY.search(line):
            rest = body[m.end():]
            if not _ITEM.search(rest) and sum(bool(_DOSEY.search(x)) for x in rest.splitlines()) < 2:
                return start + m.start()
    return end


def medication_sections(text: str) -> list[Section]:
    headers = []
    for m in _HEADER.finditer(text):
        title = _title(m)
        if title and not _INLINE.search(title):
            # A colon header counts as a section boundary only if it starts a paragraph
            # (blank line before it), is a known section, or is a medication header itself.
            prev = text.rfind("\n", 0, max(m.start() - 1, 0))
            blank_before = m.start() == 0 or text[prev + 1:m.start()].strip() == ""
            known = bool(_kind(title) or _KNOWN_END.match(title))
            # all-caps lines count only when they are known titles ("ESR/CRP" or "NONE" are not)
            if known or (blank_before and not m.group("caps")):
                headers.append((m.start(), m.end(), title))
    out = []
    for i, (hs, he, title) in enumerate(headers):
        kind = _kind(title)
        if kind:
            end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            out.append(Section(kind, title, he, _prose_break(text, he, end)))
    return out


def section_of(pos: int, sections: list[Section]) -> Section | None:
    return next((s for s in sections if s.start <= pos < s.end), None)
