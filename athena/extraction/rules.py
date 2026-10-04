"""Deterministic helpers for Branch A (README §2.1): a drug dictionary matcher,
a lab-section mask, and a rule-based attribute finder.

They complement the LLM (ensemble in the spirit of Ju et al., 2020): rules give
recall on regular medication-list syntax, the LLM handles narrative text and
status. All outputs are spans of the note text, so nothing can be invented.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from athena.config import data_path
from athena.extraction.align import Span
from athena.normalize.lexicon import load_lexicon, norm
from athena.normalize.normalizer import ABBREVIATIONS, DRUG_CLASSES

# Words that are drug names but usually appear as lab analytes or in other senses
# in discharge summaries. Tuned on n2c2 *train/val* false positives only.
AMBIGUOUS_WORDS = {
    "potassium", "calcium", "iron", "lactate", "alcohol", "magnesium", "sodium", "glucose",
    "creatinine", "creatine", "cortisol", "phosphate", "phosphorus", "chloride", "bicarbonate",
    "albumin", "protein", "water", "salt", "oxygen", "air", "ethanol", "lead", "zinc", "copper",
    "urea", "uric acid", "ammonia", "cholesterol", "triglycerides", "bilirubin", "lipase",
    "amylase", "troponin", "heparin level", "folate", "ferritin", "vitamin", "cocaine", "nicotine",
    "tobacco", "marijuana", "contrast", "sugar", "dextrose", "glycerin", "fat", "sulfur",
}
# Generic words that matched the detection vocabulary but are rarely n2c2 Drug
# mentions (found on n2c2 *train* notes: e.g. "medications" 434 FP / 24 TP).
DETECTION_STOPWORDS = {
    "medications", "medication", "meds", "home medications", "sliding scale", "sedation", "guaiac",
    "tpn", "platelets", "nebs", "nebulizers", "electrolytes", "fibrinogen", "perform",
    "bowel regimen", "repletion", "immunosuppression", "lytes", "potassium hydroxide", "thrive",
}

# Treatments n2c2 annotates as Drug although they are not interaction-checkable.
THERAPY_WORDS = {
    "ivf", "iv fluids", "prbc", "prbcs", "ffp", "platelets", "blood transfusion", "tpn",
    "chemotherapy", "chemo", "pressors", "nebs", "insulin sliding scale", "sliding scale insulin",
}

_LAB_HEADER = re.compile(
    r"^\s*(pertinent results|laboratory data|labs?|admission labs|discharge labs|lab results|"
    r"studies|imaging|micro(biology)?)\s*:", re.I | re.M)
_ANY_HEADER = re.compile(r"^\s*[A-Z][A-Za-z /&-]{2,40}:\s*$|^\s*[A-Z][A-Za-z /&-]{2,40}:\s", re.M)


def lab_sections(text: str) -> list[tuple[int, int]]:
    """Character ranges of lab/results sections (from header to next section header)."""
    out = []
    for m in _LAB_HEADER.finditer(text):
        nxt = None
        for h in _ANY_HEADER.finditer(text, m.end()):
            if not _LAB_HEADER.match(text, h.start()):
                nxt = h.start()
                break
        out.append((m.start(), nxt or len(text)))
    return out


def in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


@lru_cache(maxsize=1)
def detection_vocabulary() -> frozenset[str]:
    """Strings that mark a drug mention: KB names, RxNorm ingredient/brand names,
    abbreviations, drug classes, therapy words. Wikidata aliases are excluded here
    (too noisy for free-text detection)."""
    lex = load_lexicon()
    vocab = set(lex["canonical"])
    vocab |= {s for s, src in lex["source"].items() if src == "rxnorm"}
    rrf = data_path("raw") / "rxnorm" / "rrf" / "RXNCONSO.RRF"
    con = pd.read_csv(rrf, sep="|", header=None, dtype=str, usecols=[11, 12, 14], names=["sab", "tty", "str"])
    con = con[(con["sab"] == "RXNORM") & con["tty"].isin(["IN", "BN", "PIN"])]
    vocab |= set(con["str"].map(norm))
    vocab |= set(ABBREVIATIONS) | DRUG_CLASSES | THERAPY_WORDS
    vocab = {v for v in vocab if len(v) >= 3 and not v.isdigit()}
    return frozenset(vocab - AMBIGUOUS_WORDS - DETECTION_STOPWORDS)


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
MAX_NGRAM = 4


def dictionary_spans(text: str, skip: list[tuple[int, int]] | None = None) -> list[Span]:
    """Longest-match dictionary lookup over word n-grams (case-insensitive)."""
    vocab = detection_vocabulary()
    words = [(m.start(), m.end(), m.group(0).lower()) for m in _WORD.finditer(text)]
    out, i = [], 0
    while i < len(words):
        hit = None
        for n in range(min(MAX_NGRAM, len(words) - i), 0, -1):
            seg = words[i:i + n]
            # n-grams must not cross a line break inside the phrase
            if n > 1 and "\n" in text[seg[0][1]:seg[-1][0]]:
                continue
            key = " ".join(w for *_, w in seg)
            if key in vocab:
                hit = (seg[0][0], seg[-1][1], n)
                break
        if hit and not (skip and in_ranges(hit[0], skip)):
            out.append(Span(hit[0], hit[1], text[hit[0]:hit[1]]))
            i += hit[2]
        else:
            i += 1
    return out


# ---------------------------------------------------------------- attributes

_NUM = r"\d+(?:[.,]\d+)?(?:\s*-\s*\d+(?:[.,]\d+)?)?"
ATTR_PATTERNS: dict[str, re.Pattern] = {
    "Strength": re.compile(
        rf"(?<![\w.]){_NUM}\s*(?:mg|mcg|g|gm|grams?|units?|meq|mmol|ml|mg/ml|mcg/hr|mg/hr|units/hr|%)"
        rf"(?:\s*/\s*{_NUM}?\s*(?:ml|hr|kg|dose|patch|actuation))?(?![A-Za-z])", re.I),
    "Dosage": re.compile(
        r"(?<![\w])(?:one|two|three|four|half|1/2)\s*\(\d+(?:\.\d+)?\)", re.I),
    "Form": re.compile(
        r"(?<![A-Za-z])(?:tablets?|tabs?|capsules?|caps?|solution|suspension|syrup|patch|cream|ointment|"
        r"inhaler|spray|drops?|powder|suppository|lozenge|elixir|syringe|"
        r"tablet,?\s*(?:delayed release|extended release|chewable|rapid dissolve)(?:\s*\([^)]*\))?)(?![A-Za-z])",
        re.I),
    "Route": re.compile(
        r"(?<![A-Za-z])(?:PO|IV|IM|SC|SQ|SL|PR|subcutaneous(?:ly)?|sublingual|topical(?:ly)?|"
        r"inhalation|nebulization|intravenous(?:ly)?|intramuscular|by mouth|orally|transdermal|nasal|"
        r"ophthalmic|gtt|drip)(?![A-Za-z])", re.I),
    "Frequency": re.compile(
        r"(?<![A-Za-z])(?:Q\s?\d+\s?-?\s?\d*\s?H(?:RS?)?|QD|QOD|QHS|QAM|QPM|HS|BID|TID|QID|DAILY|"
        r"once a day|twice a day|three times a day|four times a day|every \w+ \(\d+\) hours|"
        r"every (?:morning|evening|night|day|other day)|at bedtime|nightly|weekly)(?![A-Za-z])"
        r"(?:\s*\([^()\n]{1,40}\))?(?:\s*\.?\s*as needed)?|(?<![A-Za-z])(?:PRN|as needed)(?![A-Za-z])", re.I),
    "Duration": re.compile(
        r"(?<![A-Za-z])(?:for|x)\s*(?:\d+|one|two|three|four|five|six|seven|ten|fourteen)\s*"
        r"(?:\(\d+\)\s*)?(?:more\s+)?(?:days?|weeks?|months?|doses?)(?![A-Za-z])", re.I),
}


def rule_attributes(text: str, lo: int, hi: int, taken: list[tuple[int, int]]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for t, pat in ATTR_PATTERNS.items():
        for m in pat.finditer(text, lo, hi):
            s, e = m.span()
            if any(s < b and a < e for a, b in taken):
                continue
            out.setdefault(t, []).append(Span(s, e, text[s:e]))
            taken.append((s, e))
    return out
