"""Build the interaction knowledge base (SQLite) from DrugBank extract + DDInter.

Tables
------
concepts          one row per drug concept (a group of KB names for the same drug)
names             KB name -> concept_id
interactions      one row per (concept pair, source); a < b
template_severity derived severity per DrugBank mechanism template
meta              build info and counts

Concept grouping (README §2.3, docs/limitations.md):
- reviewed RxNorm salt -> parent pairs (SALT_MERGES), and
- salt / insulin-type parentheticals used by DDInter, e.g. "insulin human (regular)".
Route-specific DDInter entries ("timolol (ophthalmic)", "lidocaine (topical)") are
deliberately NOT merged: their interaction profiles differ from systemic use.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date

import pandas as pd

from athena.config import data_path
from athena.data import ddinter, drugbank
from athena.normalize.lexicon import load_lexicon, norm

# RxNorm salt -> parent pairs (from lexicon["salt_of"]) reviewed in Batch 2b.
# Excluded on review: isosorbide mononitrate/isosorbide (different drugs),
# choline salicylate/choline (salicylate is the active moiety),
# trastuzumab deruxtecan/trastuzumab (antibody-drug conjugate),
# lutetium lu 177 dotatate & dotatate gallium ga-68 / dotatate (radiopharmaceuticals).
SALT_MERGES = {
    ("atracurium besylate", "atracurium"),
    ("clobetasol propionate", "clobetasol"),
    ("colistimethate", "colistin"),
    ("eslicarbazepine acetate", "eslicarbazepine"),
    ("fenofibric acid", "fenofibrate"),
    ("fluticasone furoate", "fluticasone"),
    ("fluticasone propionate", "fluticasone"),
    ("fosnetupitant", "netupitant"),
    ("gabapentin enacarbil", "gabapentin"),
    ("hydrocortisone butyrate", "hydrocortisone"),
    ("methscopolamine bromide", "methscopolamine"),
    ("tedizolid phosphate", "tedizolid"),
}

# Parenthetical qualifiers that denote a salt or insulin type (same systemic drug).
MERGEABLE_QUALIFIERS = {
    "potassium", "sodium", "calcium", "chloride", "sulfate",
    "regular", "isophane", "zinc", "zinc extended", "aspart", "aspart protamine", "protamine",
}
_PAREN = re.compile(r"^(?P<base>.+?) \((?P<qual>[^)]+)\)$")

SEVERITY_LEVELS = ("Major", "Moderate", "Minor", "Unknown")
# Derived severity for DrugBank-only pairs (see derive_template_severity).
MIN_OVERLAP = 10
MAJORITY = 0.5


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the shorter (parent) name as root
            if (len(ra), ra) > (len(rb), rb):
                ra, rb = rb, ra
            self.parent[rb] = ra


def build_concepts(lex: dict) -> tuple[dict[str, str], pd.DataFrame]:
    """Returns (KB name -> concept_id, concepts table)."""
    canon = set(lex["canonical"])
    uf = _UnionFind(canon)
    for salt, parent in SALT_MERGES:
        if salt in canon and parent in canon:
            uf.union(parent, salt)
    for name in canon:
        m = _PAREN.match(name)
        if m and m["qual"] in MERGEABLE_QUALIFIERS and m["base"] in canon:
            uf.union(m["base"], name)

    kag = drugbank.load_kaggle_ddi()
    in_db = {norm(n) for n in set(kag["drug_a"]) | set(kag["drug_b"])}
    in_dd = {norm(n) for n in ddinter.ddinter_drugs()["name"]}
    ids = lex["drugbank_id"]

    groups: dict[str, list[str]] = defaultdict(list)
    for name in canon:
        groups[uf.find(name)].append(name)

    rows, name_to_concept = [], {}
    for root, members in groups.items():
        members = sorted(members, key=lambda n: (n != root, n))
        db_id = ids.get(root) or next((ids[m] for m in members if m in ids), None)
        cid = db_id or "ATH:" + re.sub(r"[^a-z0-9]+", "-", root).strip("-")
        for m in members:
            name_to_concept[m] = cid
        rows.append({
            "concept_id": cid,
            "name": root,
            "drugbank_id": db_id,
            "members": json.dumps(members),
            "in_drugbank": any(m in in_db for m in members),
            "in_ddinter": any(m in in_dd for m in members),
        })
    concepts = pd.DataFrame(rows).sort_values("concept_id", ignore_index=True)
    assert concepts["concept_id"].is_unique
    return name_to_concept, concepts


def _ordered(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_interactions(name_to_concept: dict[str, str]) -> pd.DataFrame:
    kag = drugbank.load_kaggle_ddi()
    db = pd.DataFrame({
        "ca": kag["drug_a"].map(norm).map(name_to_concept),
        "cb": kag["drug_b"].map(norm).map(name_to_concept),
        "template": kag["template"],
        "description": kag["description"],
    })
    db["source"] = "drugbank"
    db["severity"] = None

    dd = ddinter.load_ddinter()
    ddf = pd.DataFrame({
        "ca": dd["drug_a"].map(norm).map(name_to_concept),
        "cb": dd["drug_b"].map(norm).map(name_to_concept),
        "template": None,
        "description": None,
        "source": "ddinter",
        "severity": dd["level"].astype(str),
    })

    df = pd.concat([db, ddf], ignore_index=True)
    assert df[["ca", "cb"]].notna().all().all(), "every source name must map to a concept"
    df = df[df["ca"] != df["cb"]]  # self-pairs created by merging salts
    ab = [_ordered(a, b) for a, b in zip(df["ca"], df["cb"])]
    df["a"], df["b"] = [x[0] for x in ab], [x[1] for x in ab]

    # One row per (pair, source). DrugBank: keep all distinct templates for the pair.
    # DDInter: after merging salts a pair may have several levels -> keep the most severe.
    db_rows = (
        df[df["source"] == "drugbank"]
        .drop_duplicates(["a", "b", "template"])
        .groupby(["a", "b"], as_index=False)
        .agg(template=("template", lambda s: json.dumps(sorted(set(s)))),
             description=("description", lambda s: json.dumps(sorted(set(s)))))
        .assign(source="drugbank", severity=None)
    )
    dd_rows = df[df["source"] == "ddinter"].copy()
    dd_rows["rank"] = dd_rows["severity"].map({s: i for i, s in enumerate(SEVERITY_LEVELS)})
    dd_rows = (
        dd_rows.sort_values("rank")
        .drop_duplicates(["a", "b"])
        .assign(template=None, description=None)[["a", "b", "template", "description", "source", "severity"]]
    )
    return pd.concat([db_rows, dd_rows], ignore_index=True)[
        ["a", "b", "source", "severity", "template", "description"]
    ]


def derive_template_severity(inter: pd.DataFrame) -> pd.DataFrame:
    """Severity for each DrugBank template, from DDInter grades of pairs in both sources.

    Rule: among overlapping pairs with a known DDInter grade (Major/Moderate/Minor),
    Major if >= 50% Major, else Minor if >= 50% Minor, else Moderate; Unknown if
    fewer than MIN_OVERLAP such pairs.
    """
    db = inter[inter["source"] == "drugbank"][["a", "b", "template"]].copy()
    db["template"] = db["template"].map(json.loads)
    db = db.explode("template")
    dd = inter[inter["source"] == "ddinter"][["a", "b", "severity"]]
    m = db.merge(dd, on=["a", "b"], how="left")

    rows = []
    for tpl, g in m.groupby("template"):
        known = g["severity"].isin(["Major", "Moderate", "Minor"])
        c = Counter(g.loc[known, "severity"])
        n = int(known.sum())
        p = {k: (c[k] / n if n else 0.0) for k in ("Major", "Moderate", "Minor")}
        if n < MIN_OVERLAP:
            sev = "Unknown"
        elif p["Major"] >= MAJORITY:
            sev = "Major"
        elif p["Minor"] >= MAJORITY:
            sev = "Minor"
        else:
            sev = "Moderate"
        rows.append({
            "template": tpl, "derived_severity": sev, "n_pairs": len(g), "n_overlap_graded": n,
            "pct_major": round(100 * p["Major"], 1), "pct_moderate": round(100 * p["Moderate"], 1),
            "pct_minor": round(100 * p["Minor"], 1),
        })
    return pd.DataFrame(rows).sort_values("n_pairs", ascending=False, ignore_index=True)


SCHEMA = """
CREATE TABLE concepts (
    concept_id TEXT PRIMARY KEY, name TEXT NOT NULL, drugbank_id TEXT,
    members TEXT NOT NULL, in_drugbank INTEGER NOT NULL, in_ddinter INTEGER NOT NULL);
CREATE TABLE names (name TEXT PRIMARY KEY, concept_id TEXT NOT NULL REFERENCES concepts(concept_id));
CREATE TABLE interactions (
    a TEXT NOT NULL REFERENCES concepts(concept_id), b TEXT NOT NULL REFERENCES concepts(concept_id),
    source TEXT NOT NULL CHECK (source IN ('drugbank', 'ddinter')),
    severity TEXT CHECK (severity IN ('Major', 'Moderate', 'Minor', 'Unknown')),
    template TEXT, description TEXT,
    PRIMARY KEY (a, b, source), CHECK (a < b));
CREATE TABLE template_severity (
    template TEXT PRIMARY KEY, derived_severity TEXT NOT NULL, n_pairs INTEGER,
    n_overlap_graded INTEGER, pct_major REAL, pct_moderate REAL, pct_minor REAL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX idx_inter_b ON interactions(b);
"""


def build_kb() -> dict:
    lex = load_lexicon()
    name_to_concept, concepts = build_concepts(lex)
    inter = build_interactions(name_to_concept)
    tsev = derive_template_severity(inter)

    path = data_path("kb_db")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        concepts.to_sql("concepts", con, if_exists="append", index=False)
        pd.Series(name_to_concept, name="concept_id").rename_axis("name").reset_index().to_sql(
            "names", con, if_exists="append", index=False)
        inter.to_sql("interactions", con, if_exists="append", index=False)
        tsev.to_sql("template_severity", con, if_exists="append", index=False)

        pairs = inter.groupby(["a", "b"])["source"].agg(frozenset)
        stats = {
            "built": date.today().isoformat(),
            "concepts": len(concepts),
            "names": len(name_to_concept),
            "pairs": len(pairs),
            "pairs_both_sources": int((pairs.map(len) == 2).sum()),
            "pairs_drugbank_only": int((pairs == frozenset({"drugbank"})).sum()),
            "pairs_ddinter_only": int((pairs == frozenset({"ddinter"})).sum()),
            "sources": "DrugBank DDI extract (Kaggle, DeepDDI benchmark); DDInter (8 ATC files)",
            "severity_rule": f"template severity from DDInter overlap: Major/Minor if >= {MAJORITY:.0%}, "
                             f"else Moderate; Unknown if < {MIN_OVERLAP} graded overlapping pairs",
        }
        con.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in stats.items()])
    return stats
