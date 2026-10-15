"""Shared pytest fixtures: load the built-in synthetic datasets."""

from pathlib import Path

import pandas as pd
import pytest

from tools import read_csv_bytes, validate_logins, validate_transactions

DATA = Path(__file__).resolve().parent.parent / "data"


def load_txns(name: str) -> pd.DataFrame:
    df, errors = read_csv_bytes((DATA / name).read_bytes())
    assert not errors
    result = validate_transactions(df)
    assert result.is_valid, result.errors
    return result.data


def load_logins(name: str) -> pd.DataFrame:
    df, errors = read_csv_bytes((DATA / name).read_bytes())
    assert not errors
    result = validate_logins(df)
    assert result.is_valid, result.errors
    return result.data


@pytest.fixture
def normal_data():
    return load_txns("normal_transactions.csv"), load_logins("normal_login_events.csv")


@pytest.fixture
def suspicious_data():
    return load_txns("suspicious_transactions.csv"), load_logins("suspicious_login_events.csv")


@pytest.fixture
def ambiguous_data():
    return load_txns("ambiguous_transactions.csv"), None
