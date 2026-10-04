"""Batch 2b: interaction knowledge base (SQLite)."""

import json
import sqlite3

import pytest

from athena.config import data_path


@pytest.fixture(scope="module")
def con():
    path = data_path("kb_db")
    if not path.exists():
        pytest.skip("run scripts/build_kb.py first")
    c = sqlite3.connect(path)
    yield c
    c.close()


def concept(con, name):
    row = con.execute("SELECT concept_id FROM names WHERE name = ?", (name,)).fetchone()
    assert row, f"{name} not in KB"
    return row[0]


def pair_rows(con, x, y):
    a, b = sorted((concept(con, x), concept(con, y)))
    return {src: (sev, tpl) for src, sev, tpl in con.execute(
        "SELECT source, severity, template FROM interactions WHERE a = ? AND b = ?", (a, b))}


def test_meta_counts(con):
    meta = dict(con.execute("SELECT key, value FROM meta"))
    assert int(meta["concepts"]) == 2421
    assert int(meta["pairs"]) == 311187
    assert int(meta["pairs_both_sources"]) == 37680


def test_integrity(con):
    assert con.execute("SELECT COUNT(*) FROM interactions WHERE a >= b").fetchone()[0] == 0
    orphan = con.execute(
        "SELECT COUNT(*) FROM interactions i LEFT JOIN concepts c ON i.a = c.concept_id WHERE c.concept_id IS NULL"
    ).fetchone()[0]
    assert orphan == 0
    assert con.execute("SELECT COUNT(*) FROM interactions WHERE source='ddinter' AND severity IS NULL").fetchone()[0] == 0


@pytest.mark.parametrize(
    "x, y, sources",
    [
        ("warfarin", "acetylsalicylic acid", {"drugbank", "ddinter"}),
        ("simvastatin", "clarithromycin", {"drugbank", "ddinter"}),
        ("heparin", "ketorolac", {"ddinter"}),  # heparin is DDInter-only (biologic)
    ],
)
def test_known_pairs_present(con, x, y, sources):
    assert sources <= set(pair_rows(con, x, y))


def test_known_severities(con):
    assert pair_rows(con, "simvastatin", "clarithromycin")["ddinter"][0] == "Major"
    assert pair_rows(con, "heparin", "ketorolac")["ddinter"][0] == "Moderate"


def test_salt_and_insulin_variants_merged(con):
    assert concept(con, "fluticasone propionate") == concept(con, "fluticasone")
    assert concept(con, "atracurium besylate") == concept(con, "atracurium")
    assert concept(con, "insulin human (regular)") == concept(con, "insulin human")
    assert concept(con, "benzylpenicillin (potassium)") == concept(con, "benzylpenicillin")


def test_route_variants_and_reviewed_exclusions_not_merged(con):
    assert concept(con, "timolol (ophthalmic)") != concept(con, "timolol")
    assert concept(con, "lidocaine (topical)") != concept(con, "lidocaine")
    assert concept(con, "isosorbide mononitrate") != concept(con, "isosorbide")
    assert concept(con, "trastuzumab deruxtecan") != concept(con, "trastuzumab")


def test_heparin_is_ddinter_only_concept(con):
    in_db, in_dd = con.execute(
        "SELECT in_drugbank, in_ddinter FROM concepts WHERE concept_id = ?", (concept(con, "heparin"),)
    ).fetchone()
    assert (in_db, in_dd) == (0, 1)


def test_template_severity_rule(con):
    rows = {t: (s, n, pm) for t, s, n, pm in con.execute(
        "SELECT template, derived_severity, n_overlap_graded, pct_major FROM template_severity")}
    assert len(rows) == 86
    for tpl, (sev, n, pct_major) in rows.items():
        if n < 10:
            assert sev == "Unknown", tpl
        elif pct_major >= 50:
            assert sev == "Major", tpl
    qtc = "The risk or severity of QTc prolongation can be increased when {A} is combined with {B}."
    assert rows[qtc][0] == "Major"


def test_drugbank_rows_keep_templates_as_json(con):
    tpl = pair_rows(con, "warfarin", "acetylsalicylic acid")["drugbank"][1]
    assert isinstance(json.loads(tpl), list) and json.loads(tpl)
