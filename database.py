"""
database.py - SQLite storage for investigation history.

Only a short summary of each investigation is stored. Uploaded datasets,
API keys and raw records are never saved. All SQL uses ? placeholders
(parameterised queries).

Note: on Streamlit Community Cloud the SQLite file lives on temporary disk
and can be wiped when the app restarts or is redeployed.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "fraudsleuth_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    created_at       TEXT NOT NULL,
    dataset_label    TEXT NOT NULL,
    n_transactions   INTEGER NOT NULL,
    risk_score       INTEGER,
    risk_category    TEXT,
    findings_summary TEXT NOT NULL,
    mode             TEXT NOT NULL,
    model            TEXT,
    status           TEXT NOT NULL
)
"""


def get_db_path() -> Path:
    """Database location; override with the FRAUDSLEUTH_DB_PATH environment variable."""
    return Path(os.getenv("FRAUDSLEUTH_DB_PATH", str(DEFAULT_DB_PATH)))


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Create the table if it does not exist yet."""
    path = Path(db_path or get_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn, conn:
        conn.execute(SCHEMA)


def summarize_findings(result: dict[str, Any]) -> str:
    """One-line description of what was found, for the history table."""
    if result.get("status") != "completed":
        return f"Not completed: {result.get('error') or 'unknown error'}"[:300]
    contributions = (result.get("risk") or {}).get("contributions", [])
    counted = [c["label"] for c in contributions if c["points"] > 0]
    if not counted:
        return "No rule-based indicators found."
    return "; ".join(counted)[:300]


def save_investigation(result: dict[str, Any], db_path: Path | str | None = None) -> None:
    """Insert one investigation summary (no raw data)."""
    path = db_path or get_db_path()
    init_db(path)
    risk = result.get("risk") or {}
    with closing(_connect(path)) as conn, conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO investigations
            (investigation_id, created_at, dataset_label, n_transactions, risk_score,
             risk_category, findings_summary, mode, model, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["investigation_id"],
                result["created_at"],
                str(result.get("dataset_label", ""))[:120],
                int(result.get("n_transactions", 0)),
                risk.get("score"),
                risk.get("category"),
                summarize_findings(result),
                result["mode"],
                result.get("model"),
                result["status"],
            ),
        )


def list_investigations(limit: int = 100, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Most recent investigations first."""
    path = db_path or get_db_path()
    init_db(path)
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT * FROM investigations ORDER BY created_at DESC, investigation_id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_investigation(investigation_id: str, db_path: Path | str | None = None) -> bool:
    """Delete one record. Returns True if a row was deleted."""
    path = db_path or get_db_path()
    init_db(path)
    with closing(_connect(path)) as conn, conn:
        cursor = conn.execute("DELETE FROM investigations WHERE investigation_id = ?", (investigation_id,))
        return cursor.rowcount > 0


def clear_history(db_path: Path | str | None = None) -> int:
    """Delete all records. Returns the number of rows deleted."""
    path = db_path or get_db_path()
    init_db(path)
    with closing(_connect(path)) as conn, conn:
        cursor = conn.execute("DELETE FROM investigations")
        return cursor.rowcount
