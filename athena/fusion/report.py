"""Fusion layer: extraction (Branch A) + verification (Branch B) -> one explainable report.

    analyze(note_text) -> Report

1. Group extracted mentions into one entry per drug; decide status from the
   note's medication sections (discharge list = active) and LLM status words.
2. Reconcile admission vs discharge lists (omitted / new / dose changed).
3. Run the deterministic checker on *active* medications.
4. Score each finding: risk = severity × evidence × min(confidence of the drugs),
   tier it (critical / review / info), and attach a plain-language explanation.

Every interaction finding is traceable to a KB record (sources, DDInter grade or
derived template evidence) and to the note text spans of both drugs. The LLM
never contributes to the risk judgement except through extraction confidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from athena.config import load_config
from athena.extraction.align import Mention
from athena.extraction.pipeline import ExtractionResult, MedicationExtractor
from athena.fusion.sections import Section, medication_sections, section_of
from athena.normalize.normalizer import Normalized
from athena.verification.checker import InteractionChecker, Medication, PairResult, PairStatus

TIER_ORDER = {"critical": 0, "review": 1, "info": 2}
HELD_WORDS = {"held", "hold", "stopped", "discontinued", "dc'd", "d/c"}
# Deterministic status cue in the same sentence as a mention (backs up the LLM status).
_STATUS_CUE = re.compile(r"\b(held|holding|stopped|discontinued|d/c'?d)\b", re.I)
_OBJECT_FILLERS = {"the", "his", "her", "their", "home", "all", "patient's", "pt's", "outpatient"}
_NEGATED_CUE = re.compile(r"\b(not|never|no longer)\s+(be\s+)?$", re.I)
_SENT_END = re.compile(r"[.;\n]\s")
# Context that makes a mention *not* a current medication (used when the note has no discharge list):
# allergy, negation, history ("allergic to aspirin", "no aspirin", "h/o amiodarone toxicity").
_NOT_CURRENT = re.compile(
    r"\b(allerg\w*|intoleran\w*|reaction to|rash (with|from|to)|anaphylaxis|"
    r"no|not on|never|denies|without|avoid|h/o|history of|previously (on|took|taking)|prior)\b[^.;\n]{0,40}$", re.I)
_ALLERGY_HEADER = re.compile(r"^[ \t]*allerg(y|ies)[^:\n]*:", re.I | re.M)


def _rule_status(text: str, start: int, end: int, other_drugs: list[tuple[int, int]]) -> str:
    """'held'/'stopped' if a cue word in the mention's sentence refers to this mention.

    A cue refers to the nearest drug *before* it in the sentence ("Simvastatin was held
    while on clarithromycin" -> simvastatin); only if no drug precedes it, to the nearest
    drug after it ("held lisinopril", "discontinued metformin")."""
    lo = max(start - 80, 0)
    b = [m.end() for m in _SENT_END.finditer(text, lo, start)]
    lo = b[-1] if b else lo
    m = _SENT_END.search(text, end, end + 80)
    hi = m.start() if m else min(end + 80, len(text))
    drugs = sorted([(s0, e0) for s0, e0 in other_drugs if lo <= s0 and e0 <= hi] + [(start, end)])
    for cue in _STATUS_CUE.finditer(text, lo, hi):
        if _NEGATED_CUE.search(text[max(cue.start() - 20, lo):cue.start()]):
            continue  # "was not held"
        before = [d for d in drugs if d[1] <= cue.start()]
        after = [d for d in drugs if d[0] >= cue.end()]
        # verb -> object: a drug right after the cue ("held aspirin", "held her home lisinopril")
        between = text[cue.end():after[0][0]].lower().split() if after else None
        if after and set(between) <= _OBJECT_FILLERS:
            target = after[0]
        else:
            target = before[-1] if before else (after[0] if after else None)
        if target == (start, end):
            w = cue.group(1).lower()
            return "held" if w.startswith("hold") or w == "held" else "stopped"
    return ""


def _not_current(text: str, start: int) -> bool:
    """True if a mention is in an allergy list or negated / historical in its sentence."""
    hdr = None
    for h in _ALLERGY_HEADER.finditer(text, 0, start):
        hdr = h
    if hdr and "\n\n" not in text[hdr.end():start] and start - hdr.end() < 400:
        return True
    b = [m.end() for m in _SENT_END.finditer(text, max(start - 80, 0), start)]
    lo = b[-1] if b else max(start - 80, 0)
    return bool(_NOT_CURRENT.search(text[lo:start]))


# ---------------------------------------------------------------- data model

@dataclass
class MentionRef:
    start: int
    end: int
    text: str
    section: str | None      # admission | discharge | None
    sources: list[str]


@dataclass
class MedEntry:
    id: str
    name: str                # display name (canonical drug name(s) or raw text)
    raw: str                 # representative text as written in the note
    norm_status: str         # mapped | ambiguous | drug_class | unmapped | non_drug
    norm_stage: str
    kb_names: list[str]
    mentions: list[MentionRef]
    attributes: dict[str, str]
    in_admission: bool
    in_discharge: bool
    text_status: str         # status word from the LLM, if any
    status: str              # active | held | stopped | mentioned
    confidence: float = 0.0
    confidence_parts: dict[str, float] = field(default_factory=dict)
    admission_attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Finding:
    id: str
    type: str                # INTERACTION | DUPLICATE_THERAPY | OMISSION | NEW_MEDICATION | DOSE_CHANGE | NOT_CHECKED | NOT_COVERED
    tier: str
    risk: float
    title: str
    meds: list[str]          # MedEntry ids
    explanation: list[str]
    evidence: dict
    needs_verification: bool = False


@dataclass
class Report:
    medications: list[MedEntry]
    findings: list[Finding]
    meta: dict

    def to_dict(self) -> dict:
        return {"medications": [asdict(m) for m in self.medications],
                "findings": [asdict(f) for f in self.findings], "meta": self.meta}

    @property
    def report_hash(self) -> str:
        return self.meta["report_hash"]


# ---------------------------------------------------------------- helpers

def _canon_attr(v: str) -> str:
    return re.sub(r"\s+", "", v.lower())


def _detection_score(sources: set[str], cfg: dict) -> float:
    llm = bool(sources & {"llm", "propagated"})
    other = sources & {"scispacy", "dictionary"}
    if llm and other:
        return cfg["llm+other"]
    if llm:
        return cfg["llm"]
    return cfg["other_multi"] if len(other) > 1 else cfg["other"]


def _group_key(norm: Normalized, text: str) -> str:
    if norm.status in ("mapped", "ambiguous") and norm.drugs:
        return "D:" + "+".join(sorted(norm.drugs))
    return "T:" + re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------- medication list

def build_medications(ex: ExtractionResult, sections: list[Section], cfg: dict) -> list[MedEntry]:
    spans = [(m.drug.start, m.drug.end) for m in ex.mentions]
    for m in ex.mentions:
        if not m.status and not section_of(m.drug.start, sections):
            # only narrative mentions: medication lists use "Sig:" lines, not status words
            others = [sp for sp in spans if sp != (m.drug.start, m.drug.end)]
            m.status = _rule_status(ex.text, m.drug.start, m.drug.end, others)
    groups: dict[str, list[tuple[Mention, Normalized, Section | None]]] = {}
    for m in ex.mentions:
        norm = ex.normalized[(m.drug.start, m.drug.end)]
        if norm.status == "non_drug":
            continue
        groups.setdefault(_group_key(norm, m.drug.text), []).append((m, norm, section_of(m.drug.start, sections)))

    has_discharge = any(s.kind == "discharge" for s in sections)
    wc, ds, ss = cfg["confidence_weights"], cfg["detection_score"], cfg["status_score"]
    entries = []
    for key, items in groups.items():
        norm = items[0][1]
        in_adm = any(s and s.kind == "admission" for *_, s in items)
        in_dis = any(s and s.kind == "discharge" for *_, s in items)

        def best(kind):
            cands = [m for m, _, s in items if (s.kind if s else None) == kind]
            return max(cands, key=lambda m: len(m.attributes), default=None)

        rep = best("discharge") or best("admission") or max((m for m, *_ in items), key=lambda m: len(m.attributes))
        adm = best("admission")
        status_words = {(m.status or "").lower() for m, *_ in items if m.status}
        text_status = next((w for w in ("stopped", "discontinued", "held", "changed", "started") if w in status_words), "")

        if has_discharge:
            if in_dis and not (status_words & HELD_WORDS and all(
                    (m.status or "").lower() in HELD_WORDS for m, _, s in items if s and s.kind == "discharge")):
                status = "active"
            elif status_words & HELD_WORDS:
                status = "stopped" if status_words & {"stopped", "discontinued", "dc'd", "d/c"} else "held"
            else:
                status = "mentioned"
            status_conf = ss["discharge_list"] if in_dis else ss["inferred"]
        else:
            # No discharge list: a drug is active unless the note holds/stops it, or every mention
            # is an allergy / negated / historical reference ("allergic to aspirin").
            if status_words & HELD_WORDS:
                status = "stopped" if status_words & {"stopped", "discontinued", "dc'd", "d/c"} else "held"
            elif all(_not_current(ex.text, m.drug.start) for m, *_ in items):
                status = "mentioned"
            else:
                status = "active"
            status_conf = ss["inferred"]

        sources = set().union(*(m.sources for m, *_ in items))
        parts = {
            "detection": _detection_score(sources, ds),
            "normalization": norm.score if norm.status == "mapped" else (0.5 if norm.status == "ambiguous" else 0.0),
            "status": status_conf,
        }
        conf = round(sum(wc[k] * v for k, v in parts.items()), 3)
        entries.append(MedEntry(
            id=key, name=" + ".join(norm.drugs) if norm.drugs else rep.drug.text,
            raw=rep.drug.text, norm_status=norm.status, norm_stage=norm.stage, kb_names=list(norm.drugs),
            mentions=[MentionRef(m.drug.start, m.drug.end, m.drug.text, s.kind if s else None, sorted(m.sources))
                      for m, _, s in items],
            attributes={t: sp[0].text for t, sp in rep.attributes.items() if sp},
            admission_attributes={t: sp[0].text for t, sp in adm.attributes.items() if sp} if adm else {},
            in_admission=in_adm, in_discharge=in_dis, text_status=text_status, status=status,
            confidence=conf, confidence_parts=parts,
        ))
    entries.sort(key=lambda e: ({"active": 0, "held": 1, "stopped": 2, "mentioned": 3}[e.status], e.name.lower()))
    return entries


# ---------------------------------------------------------------- findings

def _tier(risk: float, cfg: dict) -> str:
    t = cfg["tiers"]
    return "critical" if risk >= t["critical"] else "review" if risk >= t["review"] else "info"


def _interaction_finding(p: PairResult, entries: dict[str, MedEntry], cfg: dict) -> Finding:
    a, b = entries[p.med_a.ref], entries[p.med_b.ref]
    sev_w = cfg["severity_weight"][p.severity]
    if p.severity_basis == "derived":
        sev_w *= cfg["derived_severity_factor"]
    ev_w = cfg["evidence_weight"]["two_sources" if len(p.sources) > 1 else "one_source"]
    use_conf = cfg.get("ablation", {}).get("extraction_confidence", True)
    conf = min(a.confidence, b.confidence) if use_conf else 1.0
    risk = round(sev_w * ev_w * conf, 3)
    risk_line = f"Risk {risk:.2f} = severity {sev_w:.2f} × evidence {ev_w:.2f} × extraction confidence {conf:.2f}."
    tier = _tier(risk, cfg)
    verify = False
    if p.severity == "Major" and cfg["never_suppress_major"] and tier != "critical":
        tier, verify = "review", True

    why = [f"{a.raw!r} → {p.a.name}; {b.raw!r} → {p.b.name} (from the note text)."]
    if p.rule:
        why.append(f"High-alert rule — {p.rule.name}: {p.rule.effect} (source: {p.rule.source}).")
    if p.ddinter_level:
        why.append(f"DDInter grades this pair: {p.ddinter_level}.")
    for m in p.mechanisms[:3]:
        why.append(f"DrugBank: {m.text}")
    if p.severity_basis == "derived":
        m = p.mechanisms[0]
        why.append(f"Severity {p.severity} is derived: {m.pct_major:.0f}% of {m.n_overlap_graded} "
                   f"comparable DrugBank pairs are graded Major by DDInter.")
    why.append(risk_line)
    if verify:
        why.append("Possible Major interaction with lower extraction confidence — verify the extracted drugs.")
    return Finding(
        id=f"INT:{p.a.concept_id}|{p.b.concept_id}", type="INTERACTION", tier=tier, risk=risk,
        title=f"{p.severity} interaction: {p.a.name} + {p.b.name}", meds=[a.id, b.id], explanation=why,
        evidence={"severity": p.severity, "severity_basis": p.severity_basis, "sources": list(p.sources),
                  "ddinter_level": p.ddinter_level,
                  "mechanisms": [asdict(m) for m in p.mechanisms],
                  "rule": ({"id": p.rule.id, "name": p.rule.name, "effect": p.rule.effect, "source": p.rule.source}
                           if p.rule else None),
                  "confidence": {a.id: a.confidence_parts, b.id: b.confidence_parts}},
        needs_verification=verify,
    )


def reconciliation_findings(entries: list[MedEntry], sections: list[Section]) -> list[Finding]:
    kinds = {s.kind for s in sections}
    if not {"admission", "discharge"} <= kinds:
        return []
    out = []
    for e in entries:
        if e.norm_status in ("non_drug",):
            continue
        if e.in_admission and not e.in_discharge:
            explained = e.status in ("held", "stopped") or e.text_status in ("held", "stopped", "discontinued")
            out.append(Finding(
                id=f"OMIT:{e.id}", type="OMISSION", tier="info" if explained else "review",
                risk=0.2 if explained else 0.5,
                title=f"{'Stopped/held' if explained else 'Not on discharge list'}: {e.name}",
                meds=[e.id],
                explanation=[f"{e.raw!r} is listed on admission but not on the discharge list."] + (
                    [f"The note says it was {e.text_status or e.status}."] if explained else
                    ["No reason found in the note — confirm the omission is intended."]),
                evidence={"admission_attributes": e.admission_attributes},
            ))
        elif e.in_discharge and not e.in_admission:
            out.append(Finding(
                id=f"NEW:{e.id}", type="NEW_MEDICATION", tier="info", risk=0.1,
                title=f"New at discharge: {e.name}", meds=[e.id],
                explanation=[f"{e.raw!r} is on the discharge list but not on the admission list."],
                evidence={"attributes": e.attributes},
            ))
        elif e.in_admission and e.in_discharge:
            a, d = e.admission_attributes.get("Strength"), e.attributes.get("Strength")
            if a and d and _canon_attr(a) != _canon_attr(d):
                out.append(Finding(
                    id=f"DOSE:{e.id}", type="DOSE_CHANGE", tier="review", risk=0.4,
                    title=f"Strength changed: {e.name} {a} → {d}", meds=[e.id],
                    explanation=[f"Admission strength {a!r}, discharge strength {d!r}.",
                                 "Confirm the change is intended."],
                    evidence={"admission": a, "discharge": d},
                ))
    return out


# ---------------------------------------------------------------- entry point

# Local routes that DDInter models as separate concepts. Transdermal patches are systemic
# (fentanyl, clonidine, nitroglycerin...), so "patch" counts as topical only for local analgesics.
_LOCAL_ROUTE = [
    ("ophthalmic", re.compile(r"\b(ophthalmic|ophth|eye|eyes|ou)\b", re.I)),
    ("otic", re.compile(r"\b(otic|ear|ears)\b", re.I)),
    ("nasal", re.compile(r"\b(nasal|intranasal|nostrils?|nares)\b", re.I)),
    ("topical", re.compile(r"\b(topical|topically|tp|cream|ointment|lotion|apply|affected area|skin)\b", re.I)),
]
_LOCAL_PATCH_DRUGS = {"lidocaine", "diclofenac", "capsaicin", "menthol"}


def local_route(e: "MedEntry") -> str | None:
    """Local route of a medication from its extracted Route/Form and the drug text, if any."""
    text = " ".join([e.raw, e.attributes.get("Route", ""), e.attributes.get("Form", "")])
    for route, rx in _LOCAL_ROUTE:
        if rx.search(text):
            return route
    if re.search(r"\bpatch", text, re.I) and set(e.kb_names) & _LOCAL_PATCH_DRUGS:
        return "topical"
    return None


@lru_cache(maxsize=1)
def _default_checker() -> InteractionChecker:
    return InteractionChecker()


def analyze(text: str, extractor: MedicationExtractor | None = None,
            checker: InteractionChecker | None = None, fusion_cfg: dict | None = None) -> Report:
    full_cfg = load_config()
    cfg = fusion_cfg or full_cfg["fusion"]
    ablation = full_cfg.get("ablation", {})
    cfg.setdefault("ablation", ablation)
    extractor = extractor or MedicationExtractor("hybrid")
    checker = checker or _default_checker()

    ex = extractor.extract(text)
    use_sections = ablation.get("section_parser", True)
    sections = medication_sections(text) if use_sections else []
    entries = build_medications(ex, sections, cfg)
    by_id = {e.id: e for e in entries}

    active = [e for e in entries if e.status == "active"]
    route_aware = cfg.get("route_aware", False) and ablation.get("route_aware", True)
    check = checker.check([Medication(e.raw, "active", ref=e.id, route=local_route(e) if route_aware else None)
                           for e in active])

    findings = [_interaction_finding(p, by_id, cfg) for p in check.interactions]
    for d in check.duplicates:
        meds = [by_id[m.ref] for m in d.medications]
        conf = min(m.confidence for m in meds)
        findings.append(Finding(
            id=f"DUP:{d.concept.concept_id}", type="DUPLICATE_THERAPY", tier="review",
            risk=round(0.6 * conf, 3), title=f"Duplicate therapy: {d.concept.name}", meds=[m.id for m in meds],
            explanation=[f"{d.concept.name} is contained in: " + ", ".join(repr(m.raw) for m in meds) + "."],
            evidence={"concept": d.concept.concept_id},
        ))
    findings += reconciliation_findings(entries, sections)

    unchecked = [by_id[m.ref] for m, _ in check.unchecked]
    if unchecked:
        findings.append(Finding(
            id="NOT_CHECKED", type="NOT_CHECKED", tier="info", risk=0.0,
            title=f"{len(unchecked)} active medication(s) could not be checked", meds=[e.id for e in unchecked],
            explanation=[f"{e.raw!r}: {e.norm_status.replace('_', ' ')}" for e in unchecked]
            + ["Interactions involving these were NOT checked — review manually."],
            evidence={},
        ))
    not_cov = [p for p in check.pairs if p.status is PairStatus.NOT_COVERED]
    if not_cov:
        findings.append(Finding(
            id="NOT_COVERED", type="NOT_COVERED", tier="info", risk=0.0,
            title=f"{len(not_cov)} drug pair(s) not covered by any interaction source",
            meds=sorted({x for p in not_cov for x in (p.med_a.ref, p.med_b.ref)}),
            explanation=[f"{p.a.name} + {p.b.name}" for p in not_cov[:20]]
            + ["No source covers both drugs: absence of a record does not mean the pair is safe."],
            evidence={"pairs": [{"a": p.a.concept_id, "b": p.b.concept_id, "a_name": p.a.name, "b_name": p.b.name}
                                for p in not_cov]},
        ))
    findings.sort(key=lambda f: (TIER_ORDER[f.tier], -f.risk, f.title))

    meta = {
        "extraction_mode": ex.mode, "extraction_seconds": round(ex.seconds, 2), "warnings": ex.warnings,
        "sections": [asdict(s) for s in sections],
        "counts": {
            "medications": len(entries), "active": len(active),
            "pairs_checked": len(check.pairs),
            "interactions": len(check.interactions),
            "no_known_interaction": check.count(PairStatus.NO_KNOWN_INTERACTION),
            "not_covered": len(not_cov),
            "tiers": {t: sum(f.tier == t for f in findings) for t in TIER_ORDER},
        },
        "llm_model": getattr(extractor.llm, "model", None),
    }
    report = Report(entries, findings, meta)
    payload = json.dumps({"medications": [asdict(m) for m in entries],
                          "findings": [asdict(f) for f in findings]}, sort_keys=True)
    meta["report_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    return report
