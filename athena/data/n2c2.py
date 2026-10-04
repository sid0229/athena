"""Reader for n2c2 2018 Track 2 (BRAT standoff), straight from the zip archives.

Splits (see README §3):
    train = track2-training_data_2.zip  (265 notes)
    val   = track2-training_data_3.zip  (38 notes, author-defined hold-out)
    test  = gold-standard-test-data.zip (202 notes, official gold standard)
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePosixPath

from athena.config import data_path

SPLIT_ARCHIVES = {
    "train": "track2-training_data_2.zip",
    "val": "track2-training_data_3.zip",
    "test": "gold-standard-test-data.zip",
}

ENTITY_TYPES = ("Drug", "Strength", "Dosage", "Route", "Frequency", "Duration", "Form", "Reason", "ADE")
# Attribute types Athena extracts in v1 (Reason/ADE are out of scope, README §2.1).
V1_TYPES = ("Drug", "Strength", "Dosage", "Route", "Frequency", "Duration", "Form")


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    spans: tuple[tuple[int, int], ...]  # one or more (start, end) character offsets
    text: str

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]


@dataclass(frozen=True)
class Relation:
    id: str
    type: str  # e.g. "Strength-Drug"
    arg1: str  # attribute entity id
    arg2: str  # drug entity id


@dataclass
class Note:
    id: str
    split: str
    text: str
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def drugs(self) -> list[Entity]:
        return sorted((e for e in self.entities.values() if e.type == "Drug"), key=lambda e: e.start)

    def attributes_of(self, drug_id: str) -> list[Entity]:
        """Attribute entities linked to a drug through relations."""
        return [self.entities[r.arg1] for r in self.relations if r.arg2 == drug_id]


def parse_ann(ann: str) -> tuple[dict[str, Entity], list[Relation]]:
    entities: dict[str, Entity] = {}
    relations: list[Relation] = []
    last_t: list | None = None  # [id, type, spans, text] of the previous T line

    def flush():
        if last_t:
            entities[last_t[0]] = Entity(last_t[0], last_t[1], last_t[2], last_t[3])

    for line in ann.splitlines():
        if not line.strip():
            continue
        tag = line.split("\t", 1)[0]
        if tag.startswith("T") and tag[1:].isdigit():
            flush()
            tid, meta, text = (line.split("\t") + [""])[:3]
            etype, offsets = meta.split(" ", 1)
            spans = tuple(tuple(int(x) for x in s.split()) for s in offsets.split(";"))
            last_t = [tid, etype, spans, text]
        elif tag.startswith("R") and tag[1:].isdigit():
            flush()
            last_t = None
            rid, body = line.split("\t", 1)
            rtype, a1, a2 = body.split()
            relations.append(Relation(rid, rtype, a1.split(":", 1)[1], a2.split(":", 1)[1]))
        elif tag.startswith("#"):
            continue  # annotator notes
        elif last_t is not None:
            # Entity text that contained a newline continues on the next line.
            last_t[3] += "\n" + line
    flush()
    return entities, relations


@lru_cache(maxsize=None)
def load_split(split: str) -> tuple[Note, ...]:
    """All notes of a split, sorted by note id."""
    if split not in SPLIT_ARCHIVES:
        raise ValueError(f"split must be one of {list(SPLIT_ARCHIVES)}")
    zpath = data_path("n2c2") / SPLIT_ARCHIVES[split]
    notes = []
    with zipfile.ZipFile(zpath) as zf:
        names = {
            n for n in zf.namelist()
            if "__MACOSX" not in n and not PurePosixPath(n).name.startswith("._")
        }
        for txt_name in sorted(n for n in names if n.endswith(".txt")):
            ann_name = txt_name[:-4] + ".ann"
            text = zf.read(txt_name).decode("utf-8")
            ann = zf.read(ann_name).decode("utf-8") if ann_name in names else ""
            ents, rels = parse_ann(ann)
            notes.append(Note(PurePosixPath(txt_name).stem, split, text, ents, rels))
    return tuple(notes)


def load_all() -> dict[str, tuple[Note, ...]]:
    return {s: load_split(s) for s in SPLIT_ARCHIVES}
