"""
tools.py - Deterministic Python tools used by the FraudSleuth AI agent.

Everything in this file is plain Python + pandas. No LLM is used here.

Contents
--------
1. Configuration (AnalysisConfig) - every threshold in one place.
2. Data loading and validation for transaction and login CSV files.
3. Three investigation tools the agent may run:
     - unusual_amount_detector
     - transaction_frequency_detector
     - login_activity_checker
4. Deterministic risk scoring (the LLM can never change the score).
5. Rule-based next steps and a compact dataset summary for the LLM.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB per CSV file
MAX_ROWS = 5000  # maximum rows per CSV file

REQUIRED_TRANSACTION_COLUMNS = [
    "transaction_id",
    "account_id",
    "timestamp",
    "amount",
    "transaction_type",
    "merchant",
]
ALLOWED_TRANSACTION_TYPES = {"purchase", "transfer", "withdrawal", "deposit", "payment", "refund"}

REQUIRED_LOGIN_COLUMNS = ["event_id", "account_id", "timestamp", "status"]
ALLOWED_LOGIN_STATUSES = {"success", "failed"}

# Text that looks like an instruction to an AI. Such text is only flagged as a
# warning: free-text columns are never sent to the LLM in the first place.
SUSPICIOUS_TEXT_PATTERN = re.compile(
    r"ignore (?:all|any|previous|the above)|system prompt|you are now|disregard|"
    r"assistant:|developer message|override (?:the|your)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnalysisConfig:
    """All thresholds used by the tools. Change them here or in the sidebar."""

    min_history: int = 5  # transactions an account needs for a history comparison
    amount_multiplier: float = 3.0  # flag if amount >= multiplier x account median
    min_unusual_amount: float = 10000.0  # ...and amount is at least this much (INR)
    high_amount_threshold: float = 100000.0  # fixed "large amount" rule (INR)
    burst_window_minutes: int = 10
    burst_min_count: int = 4
    repeat_window_minutes: int = 60
    repeat_min_count: int = 3
    failed_login_threshold: int = 3
    failed_login_window_minutes: int = 30
    login_to_transaction_minutes: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of validating one CSV file."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: pd.DataFrame | None = None


@dataclass
class Finding:
    """One piece of evidence produced by a tool."""

    indicator: str
    account_id: str
    evidence_ids: list[str]
    details: dict[str, Any]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """Structured output of every investigation tool."""

    tool_name: str
    status: str  # "completed", "unavailable" or "error"
    findings: list[Finding] = field(default_factory=list)
    explanation: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "explanation": self.explanation,
            "limitations": list(self.limitations),
        }


# ---------------------------------------------------------------------------
# 2. Loading and validation
# ---------------------------------------------------------------------------


def read_csv_bytes(raw: bytes, max_bytes: int = MAX_UPLOAD_BYTES) -> tuple[pd.DataFrame | None, list[str]]:
    """Read CSV bytes into a DataFrame of strings. Returns (dataframe, errors)."""
    if raw is None or len(raw) == 0:
        return None, ["The file is empty."]
    if len(raw) > max_bytes:
        return None, [f"The file is {len(raw) / 1024 / 1024:.1f} MB; the limit is {max_bytes / 1024 / 1024:.0f} MB."]
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return None, ["The file has no data."]
    except pd.errors.ParserError as exc:
        return None, [f"The file could not be read as CSV: {exc}"]
    except UnicodeDecodeError:
        return None, ["The file is not UTF-8 text. Save it as 'CSV UTF-8' and upload again."]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df, []


def _parse_timestamps(values: pd.Series) -> tuple[pd.Series | None, str | None]:
    """Parse ISO-8601 timestamps. Returns (parsed series, error message)."""
    try:
        parsed = pd.to_datetime(values, format="ISO8601", errors="coerce")
    except (ValueError, TypeError):
        try:
            # Mixed time zones: convert everything to UTC first.
            parsed = pd.to_datetime(values, format="ISO8601", errors="coerce", utc=True)
        except (ValueError, TypeError) as exc:
            return None, f"Timestamps could not be parsed: {exc}"
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed, None


def _row_list(rows: list[int], limit: int = 10) -> str:
    shown = ", ".join(str(r) for r in rows[:limit])
    return shown + (f" and {len(rows) - limit} more" if len(rows) > limit else "")


def _common_checks(df: pd.DataFrame | None, required: list[str], label: str) -> list[str]:
    if df is None or len(df) == 0:
        return [f"The {label} file has no rows."]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"The {label} file is missing required column(s): {', '.join(missing)}."]
    if len(df) > MAX_ROWS:
        return [f"The {label} file has {len(df)} rows; the limit is {MAX_ROWS}."]
    return []


def validate_transactions(df: pd.DataFrame | None) -> ValidationResult:
    """Check a transaction table. Invalid rows are reported, never silently dropped."""
    errors = _common_checks(df, REQUIRED_TRANSACTION_COLUMNS, "transaction")
    if errors:
        return ValidationResult(False, errors)

    work = df.copy()
    for col in work.columns:
        work[col] = work[col].astype(str).str.strip()
    line = work.index + 2  # CSV line number (line 1 is the header)
    warnings: list[str] = []

    for col in ["transaction_id", "account_id", "merchant"]:
        bad = line[work[col] == ""].tolist()
        if bad:
            errors.append(f"Column '{col}' is empty on CSV line(s) {_row_list(bad)}.")

    parsed_ts, ts_error = _parse_timestamps(work["timestamp"])
    if ts_error:
        errors.append(ts_error)
    else:
        bad = line[parsed_ts.isna()].tolist()
        if bad:
            errors.append(
                f"Invalid timestamp on CSV line(s) {_row_list(bad)}. Use the format YYYY-MM-DD HH:MM:SS."
            )

    amounts = pd.to_numeric(work["amount"], errors="coerce")
    non_numeric = line[amounts.isna()].tolist()
    if non_numeric:
        errors.append(f"Amount is not a number on CSV line(s) {_row_list(non_numeric)}.")
    infinite = line[amounts.isin([math.inf, -math.inf])].tolist()
    if infinite:
        errors.append(f"Amount is not a finite number on CSV line(s) {_row_list(infinite)}.")
    not_positive = line[amounts.notna() & (amounts <= 0)].tolist()
    if not_positive:
        errors.append(
            f"Amount must be greater than zero on CSV line(s) {_row_list(not_positive)}. "
            "Use transaction_type to show direction (e.g. deposit or refund)."
        )

    types = work["transaction_type"].str.lower()
    bad = line[~types.isin(ALLOWED_TRANSACTION_TYPES)].tolist()
    if bad:
        errors.append(
            f"Unknown transaction_type on CSV line(s) {_row_list(bad)}. "
            f"Allowed values: {', '.join(sorted(ALLOWED_TRANSACTION_TYPES))}."
        )

    duplicated = work["transaction_id"][work["transaction_id"].duplicated(keep=False) & (work["transaction_id"] != "")]
    if len(duplicated):
        errors.append(f"Duplicate transaction_id value(s): {_row_list(sorted(set(duplicated)))}.")

    if work["merchant"].str.contains(SUSPICIOUS_TEXT_PATTERN).any():
        warnings.append(
            "Some merchant names contain text that looks like instructions to an AI. "
            "They are treated as plain data and are never sent to the LLM."
        )

    if errors:
        return ValidationResult(False, errors, warnings)

    clean = work.copy()
    clean["timestamp"] = parsed_ts
    clean["amount"] = amounts.astype(float)
    clean["transaction_type"] = types
    clean = clean.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    return ValidationResult(True, [], warnings, clean)


def validate_logins(df: pd.DataFrame | None) -> ValidationResult:
    """Check an optional login-event table."""
    errors = _common_checks(df, REQUIRED_LOGIN_COLUMNS, "login-event")
    if errors:
        return ValidationResult(False, errors)

    work = df.copy()
    for col in work.columns:
        work[col] = work[col].astype(str).str.strip()
    line = work.index + 2

    for col in ["event_id", "account_id"]:
        bad = line[work[col] == ""].tolist()
        if bad:
            errors.append(f"Column '{col}' is empty on login CSV line(s) {_row_list(bad)}.")

    parsed_ts, ts_error = _parse_timestamps(work["timestamp"])
    if ts_error:
        errors.append(ts_error)
    else:
        bad = line[parsed_ts.isna()].tolist()
        if bad:
            errors.append(f"Invalid timestamp on login CSV line(s) {_row_list(bad)}.")

    status = work["status"].str.lower()
    bad = line[~status.isin(ALLOWED_LOGIN_STATUSES)].tolist()
    if bad:
        errors.append(f"Login status must be 'success' or 'failed' (CSV line(s) {_row_list(bad)}).")

    duplicated = work["event_id"][work["event_id"].duplicated(keep=False) & (work["event_id"] != "")]
    if len(duplicated):
        errors.append(f"Duplicate event_id value(s): {_row_list(sorted(set(duplicated)))}.")

    if errors:
        return ValidationResult(False, errors)

    clean = work.copy()
    clean["timestamp"] = parsed_ts
    clean["status"] = status
    clean = clean.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    return ValidationResult(True, [], [], clean)


# ---------------------------------------------------------------------------
# 3. Investigation tools
# ---------------------------------------------------------------------------


def _fmt_ts(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _find_clusters(times: list[pd.Timestamp], window_minutes: int, min_count: int) -> list[tuple[int, int]]:
    """
    Find groups of at least `min_count` events that fall inside a
    `window_minutes` sliding window. Overlapping groups are merged.
    `times` must be sorted. Returns (start_index, end_index) pairs, inclusive.
    """
    window = pd.Timedelta(minutes=window_minutes)
    clusters: list[list[int]] = []
    end = 0
    for start in range(len(times)):
        end = max(end, start)
        while end + 1 < len(times) and times[end + 1] - times[start] <= window:
            end += 1
        if end - start + 1 >= min_count:
            if clusters and start <= clusters[-1][1]:
                clusters[-1][1] = max(clusters[-1][1], end)
            else:
                clusters.append([start, end])
    return [(a, b) for a, b in clusters]


def unusual_amount_detector(transactions: pd.DataFrame, config: AnalysisConfig | None = None) -> ToolResult:
    """
    Tool 1 - Unusual Amount Detector.

    Rule A (history-based): for an account with at least `min_history`
    transactions, flag an amount that is at least `amount_multiplier` times the
    median of that account's OTHER transactions AND at least `min_unusual_amount`.

    Rule B (fixed): flag any amount at or above `high_amount_threshold`.

    A large amount is only an indicator for review, never proof of fraud.
    """
    cfg = config or AnalysisConfig()
    findings: list[Finding] = []
    limitations: list[str] = []

    for account_id, group in transactions.groupby("account_id", sort=True):
        n = len(group)
        has_history = n >= cfg.min_history
        if not has_history:
            limitations.append(
                f"Account {account_id} has only {n} transaction(s) (at least {cfg.min_history} are needed), "
                "so the history-based amount rule could not be applied; only the fixed high-amount rule was used."
            )
        for idx, row in group.iterrows():
            amount = float(row["amount"])
            if has_history:
                others = group.loc[group.index != idx, "amount"]
                median = float(others.median())
                ratio = amount / median if median > 0 else math.inf
                if amount >= cfg.amount_multiplier * median and amount >= cfg.min_unusual_amount:
                    findings.append(
                        Finding(
                            indicator="unusual_amount_vs_history",
                            account_id=str(account_id),
                            evidence_ids=[str(row["transaction_id"])],
                            details={
                                "amount": round(amount, 2),
                                "account_median_of_other_transactions": round(median, 2),
                                "times_median": round(ratio, 1),
                                "rule": f">= {cfg.amount_multiplier}x median and >= {cfg.min_unusual_amount:,.0f}",
                                "timestamp": _fmt_ts(row["timestamp"]),
                            },
                            explanation=(
                                f"Transaction {row['transaction_id']} of {amount:,.2f} is {ratio:.1f} times the median "
                                f"({median:,.2f}) of the account's other {n - 1} transactions."
                            ),
                        )
                    )
            if amount >= cfg.high_amount_threshold:
                findings.append(
                    Finding(
                        indicator="high_absolute_amount",
                        account_id=str(account_id),
                        evidence_ids=[str(row["transaction_id"])],
                        details={
                            "amount": round(amount, 2),
                            "threshold": cfg.high_amount_threshold,
                            "timestamp": _fmt_ts(row["timestamp"]),
                        },
                        explanation=(
                            f"Transaction {row['transaction_id']} of {amount:,.2f} is at or above the fixed review "
                            f"threshold of {cfg.high_amount_threshold:,.0f}. A large amount alone does not indicate fraud."
                        ),
                    )
                )

    summary = (
        f"Checked {len(transactions)} transaction(s) across {transactions['account_id'].nunique()} account(s); "
        f"{len(findings)} amount indicator(s) found."
    )
    return ToolResult("unusual_amount_detector", "completed", findings, summary, limitations)


def transaction_frequency_detector(transactions: pd.DataFrame, config: AnalysisConfig | None = None) -> ToolResult:
    """
    Tool 2 - Transaction Frequency Detector.

    Rule A (burst): at least `burst_min_count` transactions from one account
    within `burst_window_minutes`.

    Rule B (repeats): at least `repeat_min_count` transactions from one account
    with the same amount and merchant within `repeat_window_minutes`.
    """
    cfg = config or AnalysisConfig()
    findings: list[Finding] = []

    for account_id, group in transactions.groupby("account_id", sort=True):
        group = group.sort_values("timestamp")
        times = group["timestamp"].tolist()
        ids = group["transaction_id"].astype(str).tolist()
        for start, end in _find_clusters(times, cfg.burst_window_minutes, cfg.burst_min_count):
            count = end - start + 1
            span = (times[end] - times[start]).total_seconds() / 60
            findings.append(
                Finding(
                    indicator="high_frequency_burst",
                    account_id=str(account_id),
                    evidence_ids=ids[start : end + 1],
                    details={
                        "transaction_count": count,
                        "first_timestamp": _fmt_ts(times[start]),
                        "last_timestamp": _fmt_ts(times[end]),
                        "span_minutes": round(span, 1),
                        "rule": f">= {cfg.burst_min_count} transactions within {cfg.burst_window_minutes} minutes",
                    },
                    explanation=(
                        f"Account {account_id} made {count} transactions in {span:.0f} minute(s) "
                        f"({_fmt_ts(times[start])} to {_fmt_ts(times[end])})."
                    ),
                )
            )

        for (amount, merchant), same in group.groupby(["amount", "merchant"], sort=True):
            same = same.sort_values("timestamp")
            s_times = same["timestamp"].tolist()
            s_ids = same["transaction_id"].astype(str).tolist()
            for start, end in _find_clusters(s_times, cfg.repeat_window_minutes, cfg.repeat_min_count):
                count = end - start + 1
                span = (s_times[end] - s_times[start]).total_seconds() / 60
                findings.append(
                    Finding(
                        indicator="repeated_identical_transactions",
                        account_id=str(account_id),
                        evidence_ids=s_ids[start : end + 1],
                        details={
                            "transaction_count": count,
                            "amount": round(float(amount), 2),
                            "merchant": str(merchant),
                            "span_minutes": round(span, 1),
                            "rule": (
                                f">= {cfg.repeat_min_count} same-amount, same-merchant transactions "
                                f"within {cfg.repeat_window_minutes} minutes"
                            ),
                        },
                        explanation=(
                            f"Account {account_id} made {count} transactions of {float(amount):,.2f} to the same "
                            f"merchant within {span:.0f} minute(s). This can be legitimate (e.g. split payments) "
                            "or a sign of duplicate or automated charges."
                        ),
                    )
                )

    summary = f"Checked transaction timing for {transactions['account_id'].nunique()} account(s); {len(findings)} frequency indicator(s) found."
    return ToolResult("transaction_frequency_detector", "completed", findings, summary, [])


def login_activity_checker(
    transactions: pd.DataFrame,
    logins: pd.DataFrame | None,
    config: AnalysisConfig | None = None,
) -> ToolResult:
    """
    Tool 3 - Optional Login Activity Checker.

    Rule A: at least `failed_login_threshold` consecutive failed logins within
    `failed_login_window_minutes`.

    Rule B: a transaction within `login_to_transaction_minutes` after a
    successful login that directly followed such a failure sequence.

    If no login data is supplied the tool says so explicitly; it never assumes
    login activity was normal.
    """
    cfg = config or AnalysisConfig()
    if logins is None or len(logins) == 0:
        return ToolResult(
            "login_activity_checker",
            "unavailable",
            [],
            "No login-event data was provided, so login activity was not checked.",
            [
                "Login evidence was unavailable. This is missing evidence, not evidence that login activity was normal."
            ],
        )

    findings: list[Finding] = []
    limitations: list[str] = []
    window = pd.Timedelta(minutes=cfg.failed_login_window_minutes)
    link_window = pd.Timedelta(minutes=cfg.login_to_transaction_minutes)

    missing = sorted(set(transactions["account_id"]) - set(logins["account_id"]))
    if missing:
        limitations.append(f"No login records were supplied for account(s): {', '.join(missing)}.")

    def record_failures(account_id: str, streak: list[pd.Series], success: pd.Series | None) -> None:
        ids = [str(e["event_id"]) for e in streak]
        details: dict[str, Any] = {
            "failed_attempts": len(streak),
            "first_failure": _fmt_ts(streak[0]["timestamp"]),
            "last_failure": _fmt_ts(streak[-1]["timestamp"]),
            "rule": f">= {cfg.failed_login_threshold} failed logins within {cfg.failed_login_window_minutes} minutes",
        }
        text = f"Account {account_id} had {len(streak)} failed login attempts between {details['first_failure']} and {details['last_failure']}"
        if success is not None:
            ids.append(str(success["event_id"]))
            details["followed_by_successful_login"] = _fmt_ts(success["timestamp"])
            text += f", followed by a successful login at {details['followed_by_successful_login']}"
        findings.append(Finding("repeated_failed_logins", account_id, ids, details, text + "."))

        if success is None:
            return
        acct_txns = transactions[transactions["account_id"] == account_id]
        after = acct_txns[
            (acct_txns["timestamp"] >= success["timestamp"]) & (acct_txns["timestamp"] <= success["timestamp"] + link_window)
        ]
        if len(after):
            txn_ids = after["transaction_id"].astype(str).tolist()
            total = float(after["amount"].sum())
            findings.append(
                Finding(
                    indicator="transaction_after_failed_logins",
                    account_id=account_id,
                    evidence_ids=[str(success["event_id"])] + txn_ids,
                    details={
                        "successful_login": _fmt_ts(success["timestamp"]),
                        "transactions_within_window": len(txn_ids),
                        "total_amount": round(total, 2),
                        "rule": f"transactions within {cfg.login_to_transaction_minutes} minutes after the login",
                    },
                    explanation=(
                        f"{len(txn_ids)} transaction(s) totalling {total:,.2f} occurred within "
                        f"{cfg.login_to_transaction_minutes} minutes after the successful login that followed the "
                        "failed attempts. This is a time association, not proof of cause."
                    ),
                )
            )

    for account_id, group in logins.groupby("account_id", sort=True):
        streak: list[pd.Series] = []
        for _, event in group.sort_values("timestamp").iterrows():
            if event["status"] == "failed":
                streak.append(event)
                # keep only failures that fit in the time window
                while streak and event["timestamp"] - streak[0]["timestamp"] > window:
                    streak.pop(0)
            else:
                if len(streak) >= cfg.failed_login_threshold:
                    record_failures(str(account_id), streak, event)
                streak = []
        if len(streak) >= cfg.failed_login_threshold:
            record_failures(str(account_id), streak, None)

    summary = f"Checked {len(logins)} login event(s) for {logins['account_id'].nunique()} account(s); {len(findings)} login indicator(s) found."
    return ToolResult("login_activity_checker", "completed", findings, summary, limitations)


# ---------------------------------------------------------------------------
# 4. Deterministic risk scoring
# ---------------------------------------------------------------------------

# Points for each indicator. Each indicator counts once per investigation,
# however many times it occurs.
INDICATOR_WEIGHTS: dict[str, int] = {
    "unusual_amount_vs_history": 25,
    "high_absolute_amount": 15,
    "high_frequency_burst": 20,
    "repeated_identical_transactions": 15,
    "repeated_failed_logins": 20,
    "transaction_after_failed_logins": 25,
}

# Indicators in the same group overlap, so only the highest one in a group counts.
INDICATOR_GROUPS: dict[str, str] = {
    "unusual_amount_vs_history": "amount",
    "high_absolute_amount": "amount",
    "high_frequency_burst": "frequency",
    "repeated_identical_transactions": "frequency",
    "repeated_failed_logins": "login_failures",
    "transaction_after_failed_logins": "login_to_transaction",
}

INDICATOR_LABELS: dict[str, str] = {
    "unusual_amount_vs_history": "Amount far above the account's usual spending",
    "high_absolute_amount": "Amount above the fixed review threshold",
    "high_frequency_burst": "Many transactions in a short time",
    "repeated_identical_transactions": "Repeated identical transactions",
    "repeated_failed_logins": "Repeated failed logins",
    "transaction_after_failed_logins": "Transaction soon after failed logins",
}

MULTI_SIGNAL_BONUS = 10  # added when 2 or more different tools found indicators

RISK_BANDS: list[tuple[int, int, str]] = [
    (0, 29, "Low"),
    (30, 59, "Medium"),
    (60, 79, "High"),
    (80, 100, "Critical"),
]


def risk_category(score: int) -> str:
    """Map a 0-100 score to its illustrative band."""
    if score < 0 or score > 100:
        raise ValueError("Score must be between 0 and 100.")
    for low, high, name in RISK_BANDS:
        if low <= score <= high:
            return name
    raise ValueError(f"No band for score {score}")  # unreachable


def calculate_risk_score(tool_results: list[ToolResult]) -> dict[str, Any]:
    """
    Combine tool findings into a 0-100 rule-based score.

    1. Each indicator present adds its weight once.
    2. Within a group of overlapping indicators only the highest weight counts.
    3. +10 if indicators came from two or more different tools.
    4. The total is capped at 100.
    """
    present: dict[str, set[str]] = {}  # indicator -> tools that found it
    for result in tool_results:
        for finding in result.findings:
            if finding.indicator in INDICATOR_WEIGHTS:
                present.setdefault(finding.indicator, set()).add(result.tool_name)

    best_in_group: dict[str, str] = {}
    for indicator in present:
        group = INDICATOR_GROUPS[indicator]
        current = best_in_group.get(group)
        if current is None or INDICATOR_WEIGHTS[indicator] > INDICATOR_WEIGHTS[current]:
            best_in_group[group] = indicator

    contributions: list[dict[str, Any]] = []
    total = 0
    for indicator in sorted(present, key=lambda i: (-INDICATOR_WEIGHTS[i], i)):
        counted = best_in_group[INDICATOR_GROUPS[indicator]] == indicator
        points = INDICATOR_WEIGHTS[indicator] if counted else 0
        total += points
        contributions.append(
            {
                "indicator": indicator,
                "label": INDICATOR_LABELS[indicator],
                "weight": INDICATOR_WEIGHTS[indicator],
                "points": points,
                "note": "Counted" if counted else f"Not added: overlaps with '{INDICATOR_LABELS[best_in_group[INDICATOR_GROUPS[indicator]]]}'",
            }
        )

    tools_with_findings = sorted({r.tool_name for r in tool_results if r.findings})
    bonus = MULTI_SIGNAL_BONUS if len(tools_with_findings) >= 2 else 0
    if bonus:
        contributions.append(
            {
                "indicator": "multiple_independent_signals",
                "label": "Indicators from two or more different tools",
                "weight": MULTI_SIGNAL_BONUS,
                "points": bonus,
                "note": f"Tools with findings: {', '.join(tools_with_findings)}",
            }
        )
    total += bonus
    uncapped = total
    score = min(total, 100)
    return {
        "score": score,
        "uncapped_score": uncapped,
        "category": risk_category(score),
        "contributions": contributions,
        "tools_with_findings": tools_with_findings,
    }


# ---------------------------------------------------------------------------
# 5. Next steps and dataset summary
# ---------------------------------------------------------------------------


def build_next_steps(tool_results: list[ToolResult], login_data_available: bool) -> list[str]:
    """Rule-based suggestions for a human reviewer, driven only by the evidence."""
    indicators = {f.indicator for r in tool_results for f in r.findings}
    limitations = [lim for r in tool_results for lim in r.limitations]
    steps: list[str] = []
    if indicators & {"unusual_amount_vs_history", "high_absolute_amount"}:
        steps.append("Compare the flagged amounts with the account holder's known spending and any supporting documents.")
    if indicators & {"high_frequency_burst", "repeated_identical_transactions"}:
        steps.append(
            "Check whether the rapid or repeated transactions are legitimate (for example split payments or "
            "subscriptions) or duplicate charges."
        )
    if "repeated_failed_logins" in indicators:
        steps.append("Review the login history and confirm with the account holder whether they made the failed attempts.")
    if "transaction_after_failed_logins" in indicators:
        steps.append("Contact the account holder to verify the transactions made shortly after the failed-login sequence.")
    if not login_data_available:
        steps.append("Request login-event records for the accounts involved; login activity could not be checked.")
    if any("history-based amount rule could not be applied" in lim for lim in limitations):
        steps.append("Obtain a longer transaction history for accounts with few transactions before drawing conclusions.")
    if not indicators:
        steps.append("No rule-based indicators were found in this data. Continue routine monitoring.")
    steps.append("A human reviewer must make the final decision. This tool does not confirm fraud.")
    return steps


def summarize_dataset(transactions: pd.DataFrame, logins: pd.DataFrame | None) -> dict[str, Any]:
    """
    Compact, numbers-only summary sent to the LLM for tool selection.
    Free-text fields (merchant names) are deliberately left out.
    """
    per_account = transactions.groupby("account_id").size()
    summary: dict[str, Any] = {
        "transaction_count": int(len(transactions)),
        "account_count": int(transactions["account_id"].nunique()),
        "date_range": [_fmt_ts(transactions["timestamp"].min()), _fmt_ts(transactions["timestamp"].max())],
        "amount_min": round(float(transactions["amount"].min()), 2),
        "amount_median": round(float(transactions["amount"].median()), 2),
        "amount_max": round(float(transactions["amount"].max()), 2),
        "transaction_types": {str(k): int(v) for k, v in transactions["transaction_type"].value_counts().items()},
        "max_transactions_per_account": int(per_account.max()),
        "min_transactions_per_account": int(per_account.min()),
        "shortest_gap_minutes_between_same_account_transactions": None,
        "login_data_available": bool(logins is not None and len(logins) > 0),
        "login_event_count": int(len(logins)) if logins is not None else 0,
        "failed_login_count": int((logins["status"] == "failed").sum()) if logins is not None and len(logins) else 0,
    }
    gaps = transactions.sort_values("timestamp").groupby("account_id")["timestamp"].diff().dropna()
    if len(gaps):
        summary["shortest_gap_minutes_between_same_account_transactions"] = round(gaps.min().total_seconds() / 60, 1)
    return summary
