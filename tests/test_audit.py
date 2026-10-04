"""Batch 6: append-only, hash-chained audit log."""

import sqlite3

import pytest

from athena.review.audit import AuditLog


@pytest.fixture
def log(tmp_path):
    return AuditLog(tmp_path / "audit.db")


def rec(log, **kw):
    base = dict(reviewer="Dr Rao", case_id="case-1", report_hash="h" * 64, finding_id="INT:a|b",
                finding_type="INTERACTION", tier="critical", title="Major interaction: a + b",
                action="confirm")
    return log.record(**(base | kw))


def test_chain_links_and_verifies(log):
    e1 = rec(log)
    e2 = rec(log, action="override", reason_code="already_managed")
    assert e1.prev_hash == "0" * 64 and e2.prev_hash == e1.row_hash
    assert log.verify() == (True, None)


def test_update_and_delete_are_rejected(log):
    rec(log)
    with sqlite3.connect(log.path) as con:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("UPDATE audit SET action='override'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            con.execute("DELETE FROM audit")


def test_tampering_is_detected(log):
    rec(log)
    rec(log)
    with sqlite3.connect(log.path) as con:
        con.execute("DROP TRIGGER audit_no_update")
        con.execute("UPDATE audit SET reviewer='someone else' WHERE id=1")
    assert log.verify() == (False, 1)


def test_override_requires_reason(log):
    with pytest.raises(ValueError, match="reason"):
        rec(log, action="override")
    with pytest.raises(ValueError, match="comment"):
        rec(log, action="override", reason_code="other")
    with pytest.raises(ValueError, match="reviewer"):
        rec(log, reviewer=" ")


def test_latest_decision_per_finding_and_report(log):
    rec(log)
    rec(log, action="override", reason_code="extraction_error")
    rec(log, report_hash="x" * 64)
    latest = log.latest_decisions("case-1", "h" * 64)
    assert latest["INT:a|b"].action == "override"
