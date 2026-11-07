"""Local API (app/server.py): path traversal, decisions tied to the viewed report, CSV escaping."""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from athena.config import ROOT


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["ATHENA_AUDIT_DB"] = str(tmp_path_factory.mktemp("audit") / "audit.db")
    sys.path.insert(0, str(ROOT / "app"))
    server = importlib.import_module("server")
    return TestClient(server.app), server


def test_spa_route_cannot_leave_dist(client):
    c, server = client
    if not server.DIST.exists():
        pytest.skip("web/dist not built")
    r = c.get("/..%2f..%2fconfigs%2fdefault.yaml")
    assert r.status_code == 404 or "paths:" not in r.text


def test_decision_needs_the_report_the_reviewer_saw(client):
    c, _ = client
    r = c.post("/api/decision", json={"report_hash": "0" * 64, "finding_id": "x", "reviewer": "r", "action": "confirm"})
    assert r.status_code == 409


def test_analyze_then_decide_round_trip(client):
    c, _ = client
    note = "Discharge Medications:\nwarfarin 5 mg daily\naspirin 81 mg daily\n"
    a = c.post("/api/analyze", json={"text": note, "mode": "rules"}).json()
    f = next(x for x in a["report"]["findings"] if x["type"] == "INTERACTION")
    d = c.post("/api/decision", json={"report_hash": a["report"]["meta"]["report_hash"], "finding_id": f["id"],
                                      "reviewer": "Dr Test", "action": "confirm"})
    assert d.status_code == 200 and f["id"] in d.json()["decisions"]


def test_csv_cells_are_formula_safe(client):
    _, server = client
    assert server._csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)" and server._csv_safe("ok") == "ok"
