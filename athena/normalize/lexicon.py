"""Build the drug lexicon: every known surface string -> canonical KB drug name(s).

Canonical drugs are the names that appear in the interaction sources
(DrugBank extract + DDInter), lower-cased. Synonyms come from:

1. Wikidata (CC0): label + aliases of the item whose DrugBank ID equals the
   canonical drug's ID. Linking by ID (not by shared strings) avoids false
   merges such as "sodium polystyrene sulfonate" -> tolevamer.
2. RxNorm prescribable content (NLM, Sep 2026): ingredient (IN) names, their
   salt forms (PIN, form_of) and brand names (BN, tradename_of), plus RxNorm
   synonyms, attached to the canonical drug whose name equals the IN name.

Also resolves DrugBank name <-> ID for the Kaggle extract (which has names
only) so it can be joined to the jcsun-00 benchmark (IDs only): Wikidata
label/alias match first, then interaction-graph matching for the rest.
"""

from __future__ import annotations

import collections
import json
import re
from functools import lru_cache

import pandas as pd

from athena.config import data_path
from athena.data import ddinter, drugbank

MIN_ALIAS_LEN = 4  # Wikidata aliases like "X", "E", "MA" are street names / noise

# Wikidata aliases found to be wrong during the Batch 2 audit: (alias regex, wrong target).
BLOCKED_ALIASES = [
    # The perflutren item carries human-albumin aliases (the microspheres are albumin-
    # shelled); albumin infusions are not perflutren.
    (re.compile(r"albumin"), "perflutren"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ---------------------------------------------------------------- sources

def _load_wikidata() -> pd.DataFrame:
    wd = pd.read_csv(data_path("raw") / "wikidata" / "drugbank_ids.csv", dtype=str)
    wd["names"] = [
        [norm(lbl)] * isinstance(lbl, str)
        + ([norm(a) for a in al.split("|")] if isinstance(al, str) else [])
        for lbl, al in zip(wd["label"], wd["aliases"])
    ]
    return wd


def _load_rxnorm() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Returns (ingredient rxcui -> name, surface string -> ingredient rxcuis)."""
    rrf = data_path("raw") / "rxnorm" / "rrf"
    con = pd.read_csv(
        rrf / "RXNCONSO.RRF", sep="|", header=None, dtype=str,
        usecols=[0, 11, 12, 14], names=["rxcui", "sab", "tty", "str"],
    )
    con = con[con["sab"] == "RXNORM"]
    rel = pd.read_csv(
        rrf / "RXNREL.RRF", sep="|", header=None, dtype=str,
        usecols=[0, 4, 7, 10], names=["rxcui1", "rxcui2", "rela", "sab"],
    )
    # Row reads "rxcui2 <rela> rxcui1", e.g. PIN form_of IN, BN tradename_of IN.
    rel = rel[(rel["sab"] == "RXNORM") & rel["rela"].isin(["form_of", "tradename_of"])]

    ins = con[con["tty"] == "IN"]
    in_name = dict(zip(ins["rxcui"], ins["str"].map(norm)))
    to_in: dict[str, set[str]] = collections.defaultdict(set)
    for c in in_name:
        to_in[c].add(c)
    for r1, r2 in zip(rel["rxcui1"], rel["rxcui2"]):
        if r1 in in_name:
            to_in[r2].add(r1)

    surface: dict[str, set[str]] = collections.defaultdict(set)
    terms = con[con["tty"].isin(["IN", "PIN", "BN", "SY", "TMSY"])]
    for rxcui, s in zip(terms["rxcui"], terms["str"]):
        for c in to_in.get(rxcui, ()):
            surface[norm(s)].add(c)
    return in_name, surface


# ---------------------------------------------------------------- name <-> DrugBank ID

def resolve_drugbank_ids(wd: pd.DataFrame) -> dict[str, str]:
    """Kaggle drug name -> DrugBank ID, validated against the benchmark graph."""
    kag = drugbank.load_kaggle_ddi()
    bench = drugbank.load_benchmark_ddi()
    names = sorted(set(kag["drug_a"]) | set(kag["drug_b"]))
    bench_ids = set(bench["id_a"]) | set(bench["id_b"])

    by_label, by_alias = collections.defaultdict(set), collections.defaultdict(set)
    for db, lbl, nms in zip(wd["db"], wd["label"], wd["names"]):
        if isinstance(lbl, str):
            by_label[norm(lbl)].add(db)
        for a in nms:
            if len(a) >= MIN_ALIAS_LEN:
                by_alias[a].add(db)

    name2id = {}
    for n in names:
        key = norm(n)
        for index in (by_label, by_alias):
            hits = index.get(key, set()) & bench_ids
            if len(hits) == 1:
                name2id[n] = next(iter(hits))
                break
    claimed = collections.Counter(name2id.values())
    name2id = {n: i for n, i in name2id.items() if claimed[i] == 1}

    # Graph matching: an unmapped name's mapped neighbours must be (almost) exactly
    # the neighbours of one free benchmark ID.
    k_nb, b_nb = collections.defaultdict(set), collections.defaultdict(set)
    for a, b in zip(kag["drug_a"], kag["drug_b"]):
        k_nb[a].add(b)
        k_nb[b].add(a)
    for a, b in zip(bench["id_a"], bench["id_b"]):
        b_nb[a].add(b)
        b_nb[b].add(a)
    free = bench_ids - set(name2id.values())
    for n in names:
        if n in name2id:
            continue
        nb = {name2id[m] for m in k_nb[n] if m in name2id}
        if len(nb) < 3:
            continue
        scored = sorted(((len(nb & b_nb[i]) / len(nb | (b_nb[i] - free)), i) for i in free), reverse=True)
        if scored and scored[0][0] >= 0.9 and (len(scored) == 1 or scored[0][0] - scored[1][0] > 0.2):
            name2id[n] = scored[0][1]
            free.discard(scored[0][1])
    return name2id


# ---------------------------------------------------------------- lexicon

def build_lexicon() -> dict:
    wd = _load_wikidata()
    in_name, rx_surface = _load_rxnorm()

    kag = drugbank.load_kaggle_ddi()
    canon = {norm(n) for n in set(kag["drug_a"]) | set(kag["drug_b"])}
    canon |= {norm(n) for n in ddinter.ddinter_drugs()["name"]}

    name2id = resolve_drugbank_ids(wd)
    canon_id = {norm(n): i for n, i in name2id.items()}
    # DDInter-only drugs: take the DrugBank ID of a unique Wikidata label match.
    label_ids = collections.defaultdict(set)
    for db, lbl in zip(wd["db"], wd["label"]):
        if isinstance(lbl, str):
            label_ids[norm(lbl)].add(db)
    for c in canon:
        if c not in canon_id and len(label_ids.get(c, ())) == 1:
            canon_id[c] = next(iter(label_ids[c]))
    id_names = collections.defaultdict(list)
    for db, nms in zip(wd["db"], wd["names"]):
        id_names[db].extend(nms)

    index: dict[str, set[str]] = collections.defaultdict(set)
    source: dict[str, str] = {}

    def add(surface: str, target: str, src: str):
        if surface not in index or target not in index[surface]:
            index[surface].add(target)
            source.setdefault(surface, src)

    # An alias that is the name of an RxNorm ingredient *other than the item's own
    # RxNorm ingredient* is a loose Wikidata link (e.g. the perflutren item lists
    # "albumin human") — skip it. "aspirin" on acetylsalicylic acid is kept because
    # that item's RxCUI is the aspirin ingredient.
    rxcui_by_name = collections.defaultdict(set)
    for rx, nm in in_name.items():
        rxcui_by_name[nm].add(rx)
    db_rxcuis = collections.defaultdict(set)
    for db, rx in zip(wd["db"], wd["rxcui"]):
        if isinstance(rx, str):
            db_rxcuis[db].add(rx)

    def loose_alias(alias: str, db: str) -> bool:
        alias_rx = rxcui_by_name.get(alias)
        return bool(alias_rx) and not (alias_rx & db_rxcuis.get(db, set()))

    for c in canon:
        add(c, c, "kb")
    for c, db in canon_id.items():
        for a in id_names.get(db, ()):
            if len(a) < MIN_ALIAS_LEN or a in canon:
                continue
            if a != c and loose_alias(a, db):
                continue
            if any(c == target and pat.search(a) for pat, target in BLOCKED_ALIASES):
                continue
            add(a, c, "wikidata")

    # Where the RxNorm ingredient itself is not a KB drug but exactly one of its
    # salt/brand forms is (e.g. "megestrol" -> KB "megestrol acetate"), use that form.
    in_canon_forms = collections.defaultdict(set)
    for surface, rxcuis in rx_surface.items():
        if surface in canon:
            for rx in rxcuis:
                in_canon_forms[rx].add(surface)

    def rx_target(rx: str) -> str | None:
        name = in_name[rx]
        if name in canon:
            return name
        if name in index and len(index[name]) == 1:
            return next(iter(index[name]))
        if len(in_canon_forms.get(rx, ())) == 1:
            return next(iter(in_canon_forms[rx]))
        return None

    for surface, rxcuis in rx_surface.items():
        if surface in canon:
            continue
        for rx in rxcuis:
            if target := rx_target(rx):
                add(surface, target, "rxnorm")

    return {
        "canonical": sorted(canon),
        "drugbank_id": dict(sorted(canon_id.items())),
        "kaggle_name_to_id": dict(sorted(name2id.items())),
        "index": {s: sorted(t) for s, t in sorted(index.items())},
        "source": source,
    }


LEXICON_FILE = "drug_lexicon.json"


def save_lexicon(lex: dict) -> None:
    out = data_path("processed") / LEXICON_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lex))


@lru_cache(maxsize=None)
def load_lexicon() -> dict:
    path = data_path("processed") / LEXICON_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python scripts/build_lexicon.py`")
    return json.loads(path.read_text())
