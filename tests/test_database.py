"""Tests for SQLite investigation history (uses a temporary database file)."""

import sqlite3

from agent import MODE_RULE_BASED, run_investigation
from database import clear_history, delete_investigation, init_db, list_investigations, save_investigation


def test_database_is_created_automatically(tmp_path):
    db = tmp_path / "sub" / "history.db"
    init_db(db)
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "investigations" in tables


def test_save_list_delete_and_clear(tmp_path, suspicious_data, normal_data):
    db = tmp_path / "history.db"
    r1 = run_investigation(*suspicious_data, "Suspicious demo", MODE_RULE_BASED)
    r2 = run_investigation(*normal_data, "Normal demo", MODE_RULE_BASED)
    save_investigation(r1, db)
    save_investigation(r2, db)

    rows = list_investigations(db_path=db)
    assert len(rows) == 2
    saved = {r["investigation_id"]: r for r in rows}[r1["investigation_id"]]
    assert saved["risk_score"] == 100
    assert saved["risk_category"] == "Critical"
    assert saved["mode"] == "rule_based"
    assert saved["n_transactions"] == 17
    assert "Repeated failed logins" in saved["findings_summary"]
    # only summary columns are stored - no raw transactions
    assert set(saved) == {
        "investigation_id", "created_at", "dataset_label", "n_transactions", "risk_score",
        "risk_category", "findings_summary", "mode", "model", "status",
    }

    assert delete_investigation(r1["investigation_id"], db) is True
    assert delete_investigation(r1["investigation_id"], db) is False
    assert len(list_investigations(db_path=db)) == 1
    assert clear_history(db) == 1
    assert list_investigations(db_path=db) == []


def test_parameterised_queries_handle_quotes_safely(tmp_path, normal_data):
    db = tmp_path / "history.db"
    result = run_investigation(*normal_data, "x'); DROP TABLE investigations; --", MODE_RULE_BASED)
    save_investigation(result, db)
    rows = list_investigations(db_path=db)
    assert rows[0]["dataset_label"] == "x'); DROP TABLE investigations; --"


def test_failed_investigation_is_stored_with_reason(tmp_path, normal_data):
    db = tmp_path / "history.db"
    failed = run_investigation(*normal_data, "Normal demo", "llm", llm=None)
    save_investigation(failed, db)
    row = list_investigations(db_path=db)[0]
    assert row["status"] == "failed"
    assert row["risk_score"] is None
    assert row["findings_summary"].startswith("Not completed")
