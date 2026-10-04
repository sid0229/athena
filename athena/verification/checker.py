"""Branch B: deterministic pairwise interaction checker (README §2.3).

Given a medication list, every pair of *active* drug concepts gets exactly one result:

    INTERACTION           found in >= 1 source (severity, basis, sources, mechanisms)
    NO_KNOWN_INTERACTION  both drugs are covered by a common source, and it has no record
    NOT_COVERED           no source covers both drugs -> "unknown", never "safe"

Also reported:
    DUPLICATE_THERAPY     the same drug concept appears in two different medications
                          (e.g. Percocet + Tylenol -> acetaminophen twice)
    unchecked medications drug classes, unmapped and ambiguous names (shown, not checked)

Severity of an INTERACTION:
    1. DDInter grade if Major/Moderate/Minor          basis "ddinter"
    2. else most severe derived DrugBank template     basis "derived"
    3. else DDInter "Unknown"                         basis "ddinter"
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from athena.config import data_path
from athena.normalize.normalizer import DrugNormalizer, Normalized

SEVERITY_RANK = {"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3}


class PairStatus(str, Enum):
    INTERACTION = "INTERACTION"
    NO_KNOWN_INTERACTION = "NO_KNOWN_INTERACTION"
    NOT_COVERED = "NOT_COVERED"


@dataclass(frozen=True)
class Medication:
    """A medication as entered or extracted. Only status 'active' is checked."""
    name: str
    status: str = "active"   # active | held | stopped | unknown
    ref: str | None = None   # caller's id (e.g. extracted entity id), passed through


@dataclass(frozen=True)
class Concept:
    concept_id: str
    name: str
    in_drugbank: bool
    in_ddinter: bool

    @property
    def sources(self) -> frozenset[str]:
        return frozenset(s for s, ok in (("drugbank", self.in_drugbank), ("ddinter", self.in_ddinter)) if ok)


@dataclass(frozen=True)
class Mechanism:
    template: str
    text: str                 # original DrugBank sentence (drug order as in the source)
    derived_severity: str
    n_overlap_graded: int
    pct_major: float


@dataclass
class PairResult:
    a: Concept
    b: Concept
    med_a: Medication
    med_b: Medication
    status: PairStatus
    severity: str | None = None          # Major | Moderate | Minor | Unknown
    severity_basis: str | None = None    # ddinter | derived
    sources: tuple[str, ...] = ()
    ddinter_level: str | None = None
    mechanisms: tuple[Mechanism, ...] = ()
    checked_sources: tuple[str, ...] = ()  # sources that cover both drugs

    @property
    def key(self) -> tuple[str, str]:
        return (self.a.concept_id, self.b.concept_id)


@dataclass
class Duplicate:
    concept: Concept
    medications: tuple[Medication, ...]


@dataclass
class CheckReport:
    medications: list[tuple[Medication, Normalized]]
    pairs: list[PairResult] = field(default_factory=list)
    duplicates: list[Duplicate] = field(default_factory=list)
    unchecked: list[tuple[Medication, Normalized]] = field(default_factory=list)
    inactive: list[Medication] = field(default_factory=list)

    @property
    def interactions(self) -> list[PairResult]:
        found = [p for p in self.pairs if p.status is PairStatus.INTERACTION]
        return sorted(found, key=lambda p: (SEVERITY_RANK[p.severity], p.a.name, p.b.name))

    def count(self, status: PairStatus) -> int:
        return sum(p.status is status for p in self.pairs)


class InteractionKB:
    """Read-only access to data/processed/interactions.db."""

    def __init__(self, path=None):
        path = path or data_path("kb_db")
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `python scripts/build_kb.py`")
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self.concepts = {
            cid: Concept(cid, name, bool(db), bool(dd))
            for cid, name, db, dd in self.con.execute(
                "SELECT concept_id, name, in_drugbank, in_ddinter FROM concepts")
        }
        self.name_to_concept = dict(self.con.execute("SELECT name, concept_id FROM names"))
        self.template_severity = {
            t: (s, n, pm) for t, s, n, pm in self.con.execute(
                "SELECT template, derived_severity, n_overlap_graded, pct_major FROM template_severity")
        }

    def concept_for(self, kb_name: str) -> Concept:
        return self.concepts[self.name_to_concept[kb_name]]

    def lookup(self, c1: str, c2: str) -> list[tuple[str, str | None, str | None, str | None]]:
        """Rows (source, severity, templates_json, descriptions_json) for an unordered pair."""
        a, b = sorted((c1, c2))
        return self.con.execute(
            "SELECT source, severity, template, description FROM interactions WHERE a = ? AND b = ?",
            (a, b),
        ).fetchall()


class InteractionChecker:
    def __init__(self, kb: InteractionKB | None = None, normalizer: DrugNormalizer | None = None):
        self.kb = kb or InteractionKB()
        self.normalizer = normalizer or DrugNormalizer()

    def check(self, medications: list[Medication | str]) -> CheckReport:
        meds = [m if isinstance(m, Medication) else Medication(m) for m in medications]
        report = CheckReport(medications=[(m, self.normalizer.normalize(m.name)) for m in meds])

        # (medication index, concept) for every checkable active medication
        entries: list[tuple[int, Concept]] = []
        for i, (med, norm) in enumerate(report.medications):
            if med.status != "active":
                report.inactive.append(med)
                continue
            if norm.status == "non_drug":
                continue
            if norm.status != "mapped":
                report.unchecked.append((med, norm))
                continue
            for d in norm.drugs:
                entries.append((i, self.kb.concept_for(d)))

        # Duplicate therapy: same concept from two different medications.
        by_concept: dict[str, list[int]] = {}
        for i, c in entries:
            by_concept.setdefault(c.concept_id, []).append(i)
        for cid, idxs in by_concept.items():
            if len(set(idxs)) > 1:
                report.duplicates.append(Duplicate(
                    self.kb.concepts[cid],
                    tuple(report.medications[i][0] for i in sorted(set(idxs)))))

        # Pairs: distinct concepts from different medications, each pair once.
        seen: set[tuple[str, str]] = set()
        for (i, c1), (j, c2) in itertools.combinations(entries, 2):
            if i == j or c1.concept_id == c2.concept_id:
                continue  # same co-formulated product, or duplicate (reported above)
            if c1.concept_id > c2.concept_id:
                (i, c1), (j, c2) = (j, c2), (i, c1)
            if (c1.concept_id, c2.concept_id) in seen:
                continue
            seen.add((c1.concept_id, c2.concept_id))
            report.pairs.append(self._check_pair(c1, c2, report.medications[i][0], report.medications[j][0]))
        return report

    def _check_pair(self, a: Concept, b: Concept, med_a: Medication, med_b: Medication) -> PairResult:
        rows = self.kb.lookup(a.concept_id, b.concept_id)
        checked = tuple(sorted(a.sources & b.sources))
        if not rows:
            status = PairStatus.NO_KNOWN_INTERACTION if checked else PairStatus.NOT_COVERED
            return PairResult(a, b, med_a, med_b, status, checked_sources=checked)

        ddinter_level, mechanisms = None, []
        for source, severity, templates, descriptions in rows:
            if source == "ddinter":
                ddinter_level = severity
            else:
                for t, text in zip(json.loads(templates), json.loads(descriptions)):
                    sev, n, pm = self.kb.template_severity[t]
                    mechanisms.append(Mechanism(t, text, sev, n, pm))
        mechanisms.sort(key=lambda m: SEVERITY_RANK[m.derived_severity])

        if ddinter_level in ("Major", "Moderate", "Minor"):
            severity, basis = ddinter_level, "ddinter"
        elif mechanisms:
            severity, basis = mechanisms[0].derived_severity, "derived"
        else:
            severity, basis = "Unknown", "ddinter"

        return PairResult(
            a, b, med_a, med_b, PairStatus.INTERACTION,
            severity=severity, severity_basis=basis,
            sources=tuple(sorted(s for s, *_ in rows)),
            ddinter_level=ddinter_level, mechanisms=tuple(mechanisms), checked_sources=checked,
        )


@lru_cache(maxsize=1)
def default_checker() -> InteractionChecker:
    return InteractionChecker()
