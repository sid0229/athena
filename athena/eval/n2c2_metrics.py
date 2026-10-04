"""n2c2 2018 Track 2 style scoring for medication entities and attribute->drug relations.

strict  : same type and identical character extent
lenient : same type and overlapping extent (one-to-one greedy matching)
The extent of a discontiguous gold entity is (first start, last end).
Scores are per type and micro-averaged over the evaluated types.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from athena.data.n2c2 import V1_TYPES, Note
from athena.extraction.align import Mention

Ent = tuple[str, int, int]            # (type, start, end)
Rel = tuple[str, int, int, int, int]  # (attr type, attr start, attr end, drug start, drug end)


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def p(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def r(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0

    def __iadd__(self, o: "PRF"):
        self.tp += o.tp
        self.fp += o.fp
        self.fn += o.fn
        return self


@dataclass
class Scores:
    strict: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    lenient: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    rel_strict: PRF = field(default_factory=PRF)
    rel_lenient: PRF = field(default_factory=PRF)

    def micro(self, mode: str = "strict", types=V1_TYPES) -> PRF:
        table = self.strict if mode == "strict" else self.lenient
        out = PRF()
        for t in types:
            out += table[t]
        return out

    def as_dict(self) -> dict:
        row = lambda x: {"p": round(x.p, 4), "r": round(x.r, 4), "f1": round(x.f1, 4),  # noqa: E731
                         "tp": x.tp, "fp": x.fp, "fn": x.fn}
        return {
            "strict": {t: row(self.strict[t]) for t in V1_TYPES} | {"micro": row(self.micro("strict"))},
            "lenient": {t: row(self.lenient[t]) for t in V1_TYPES} | {"micro": row(self.micro("lenient"))},
            "relations": {"strict": row(self.rel_strict), "lenient": row(self.rel_lenient)},
        }


def gold_entities(note: Note, types=V1_TYPES) -> list[Ent]:
    return [(e.type, e.start, e.end) for e in note.entities.values() if e.type in types]


def gold_relations(note: Note, types=V1_TYPES) -> list[Rel]:
    out = []
    for r in note.relations:
        a, d = note.entities[r.arg1], note.entities[r.arg2]
        if a.type in types and a.type != "Drug":
            out.append((a.type, a.start, a.end, d.start, d.end))
    return out


def predicted(mentions: list[Mention]) -> tuple[list[Ent], list[Rel]]:
    ents, rels = [], []
    for m in mentions:
        ents.append(("Drug", m.drug.start, m.drug.end))
        for t, spans in m.attributes.items():
            for s in spans:
                ents.append((t, s.start, s.end))
                rels.append((t, s.start, s.end, m.drug.start, m.drug.end))
    return sorted(set(ents)), sorted(set(rels))


def _match(gold: list, pred: list, same) -> PRF:
    used = set()
    tp = 0
    for g in gold:
        for i, p in enumerate(pred):
            if i not in used and same(g, p):
                used.add(i)
                tp += 1
                break
    return PRF(tp, len(pred) - tp, len(gold) - tp)


def _ov(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def score_note(note: Note, mentions: list[Mention], scores: Scores | None = None) -> Scores:
    scores = scores or Scores()
    g_ents, g_rels = gold_entities(note), gold_relations(note)
    p_ents, p_rels = predicted(mentions)
    for t in V1_TYPES:
        g = [e for e in g_ents if e[0] == t]
        p = [e for e in p_ents if e[0] == t]
        scores.strict[t] += _match(g, p, lambda a, b: a == b)
        scores.lenient[t] += _match(g, p, lambda a, b: _ov(a[1], a[2], b[1], b[2]))
    scores.rel_strict += _match(g_rels, p_rels, lambda a, b: a == b)
    scores.rel_lenient += _match(
        g_rels, p_rels,
        lambda a, b: a[0] == b[0] and _ov(a[1], a[2], b[1], b[2]) and _ov(a[3], a[4], b[3], b[4]))
    return scores
