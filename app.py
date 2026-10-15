"""
app.py - Streamlit frontend for VerifAI.

Run locally:  python -m streamlit run app.py

LLM requests happen ONLY when the user clicks "Investigate Transactions".
Opening the app, and previewing data never call Groq.
"""

from __future__ import annotations

import hashlib
import html
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import MODE_LLM, MODE_RULE_BASED, TOOL_DESCRIPTIONS, run_investigation
from database import clear_history, delete_investigation, list_investigations, save_investigation
from groq_service import GroqService, LLMServiceError, get_api_key, get_model_name
from report_generator import build_findings_csv, build_markdown_report
from tools import (
    ALLOWED_LOGIN_STATUSES,
    ALLOWED_TRANSACTION_TYPES,
    INDICATOR_GROUPS,
    INDICATOR_LABELS,
    INDICATOR_WEIGHTS,
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    MULTI_SIGNAL_BONUS,
    RISK_BANDS,
    AnalysisConfig,
    ValidationResult,
    read_csv_bytes,
    validate_logins,
    validate_transactions,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
BUILT_IN_DATASETS: dict[str, tuple[str, str | None]] = {
    "Normal transactions": ("normal_transactions.csv", "normal_login_events.csv"),
    "Several suspicious indicators": ("suspicious_transactions.csv", "suspicious_login_events.csv"),
    "Ambiguous activity, limited evidence": ("ambiguous_transactions.csv", None),
}
MODE_OPTIONS = {"LLM Agent Mode (Groq)": MODE_LLM, "Rule-Based Demo Mode (no LLM)": MODE_RULE_BASED}
LLM_ERROR_KINDS = {"auth", "permission", "rate_limit", "model_unavailable", "bad_request", "timeout", "network", "api_error", "invalid_response", "missing_key"}

LOGO_SVG = (Path(__file__).resolve().parent / "assets" / "verifai_logo.svg").read_text(encoding="utf-8")
LOGO_PNG = Path(__file__).resolve().parent / "assets" / "verifai_logo.png"

st.set_page_config(page_title="VerifAI", page_icon=str(LOGO_PNG), layout="wide")

# Small CSS additions; the main palette lives in .streamlit/config.toml
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 1180px; }
    h1, h2, h3 { color: #4B2E24; letter-spacing: -0.01em; }
    .fs-brand { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.4rem; }
    .fs-brand svg { width: 64px; height: 64px; flex-shrink: 0; }
    .fs-brand h1 { margin: 0; padding: 0; font-variant-ligatures: none; font-feature-settings: "liga" 0, "clig" 0; letter-spacing: 0.02em; }
    .fs-disclaimer { border-left: 4px solid #9B1C31; background: #F5EBDD; padding: 0.75rem 1rem;
                     border-radius: 0.3rem; color: #4B2E24; margin: 0.8rem 0 1.2rem 0; max-width: 90ch; }
    .fs-case { display: flex; gap: 2rem; align-items: center; flex-wrap: wrap; background: #F5EBDD;
               border: 1px solid #D8C3A5; border-radius: 0.5rem; padding: 1.2rem 1.5rem; margin: 0.5rem 0 1rem 0; }
    .fs-score { font-family: 'Spectral', serif; font-size: 3.6rem; line-height: 1; color: #9B1C31; font-weight: 700; }
    .fs-score small { font-size: 1.2rem; color: #79665A; font-weight: 500; }
    .fs-band { display: inline-block; border: 1.5px solid #741426; color: #741426; background: #FFFCF5;
               border-radius: 999px; padding: 0.15rem 0.8rem; font-weight: 600; margin-top: 0.35rem; }
    .fs-case-meta { color: #79665A; line-height: 1.7; }
    .fs-case-meta b { color: #4B2E24; }
    .fs-mode { display: inline-block; border-radius: 999px; padding: 0.1rem 0.7rem; font-size: 0.9rem;
               border: 1px solid #D8C3A5; background: #FFFCF5; color: #4B2E24; }
    .fs-muted { color: #79665A; font-size: 0.92rem; }
    div[data-testid="stMetric"] { background: #F5EBDD; border: 1px solid #D8C3A5; border-radius: 0.4rem; padding: 0.6rem 0.9rem; }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover { background-color: #741426; border-color: #741426; color: #FFFCF5; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=20)
def load_and_validate(raw: bytes, kind: str) -> ValidationResult:
    """Read and validate CSV bytes. Cached, and never calls the LLM."""
    df, errors = read_csv_bytes(raw, MAX_UPLOAD_BYTES)
    if errors:
        return ValidationResult(False, errors)
    return validate_transactions(df) if kind == "transactions" else validate_logins(df)


def fingerprint(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else json.dumps(part, sort_keys=True, default=str).encode())
        digest.update(b"|")
    return digest.hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value))


def llm_text(value: object) -> str:
    """Show LLM text as written: escape $ so Streamlit does not treat it as maths."""
    return str(value).replace("$", "\\$")


def show_validation(result: ValidationResult, label: str) -> None:
    if result.is_valid:
        st.success(f"{label}: {len(result.data)} rows passed validation.")
    else:
        st.error(f"{label} failed validation. Fix the data file:\n\n" + "\n".join(f"- {e}" for e in result.errors))
    for w in result.warnings:
        st.warning(w)


def trace_table(trace: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(trace).rename(columns={"step": "Step", "node": "Node", "actor": "Actor", "status": "Status", "detail": "Detail"})
    return table.set_index("Step")


def preview(df: pd.DataFrame) -> None:
    shown = df.head(50).copy()
    if "timestamp" in shown:
        shown["timestamp"] = shown["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(shown, hide_index=True, width="stretch")
    if len(df) > 50:
        st.caption(f"Showing the first 50 of {len(df)} rows.")


# ---------------------------------------------------------------------------
# Sidebar: operating mode
# ---------------------------------------------------------------------------

api_key = get_api_key()
model_name = get_model_name()

with st.sidebar:
    st.header("Operating mode")
    mode_label = st.radio("Choose how the agent runs", list(MODE_OPTIONS), key="mode_label")
    mode = MODE_OPTIONS[mode_label]
    if mode == MODE_LLM:
        if not api_key:
            st.warning("No GROQ_API_KEY found, so LLM Agent Mode is unavailable. Add the key (see README) or choose Rule-Based Demo Mode.")
    else:
        st.info("Rule-Based Demo Mode: fixed Python rules choose the tools and write the summary. No LLM is used.")

cfg = AnalysisConfig()  # fixed default thresholds

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="fs-brand">{LOGO_SVG}<h1>VerifAI</h1></div>',
    unsafe_allow_html=True,
)



tab_investigate, tab_history, tab_how = st.tabs(["Investigate", "History", "How it works"])

# ---------------------------------------------------------------------------
# Investigate tab
# ---------------------------------------------------------------------------


def render_result(result: dict) -> None:
    if result["status"] != "completed":
        st.error(f"The investigation did not finish: {result.get('error')}")
        if result.get("error_kind") in LLM_ERROR_KINDS:
            st.info(
                "No rule-based result was substituted. To analyse this data without the LLM, choose "
                "Rule-Based Demo Mode in the sidebar and click Investigate Transactions again."
            )
        with st.expander("Agent execution trace", expanded=True):
            st.table(trace_table(result["trace"]))
        return

    risk = result["risk"]
    mode_text = f"LLM Agent Mode ({result['model']} via Groq)" if result["mode"] == MODE_LLM else "Rule-Based Demo Mode (no LLM used)"
    st.markdown(
        f"""
        <div class="fs-case">
          <div>
            <div class="fs-score">{risk['score']}<small> / 100</small></div>
            <div class="fs-band">Risk category: {esc(risk['category'])}</div>
          </div>
          <div class="fs-case-meta">
            <b>{esc(result['investigation_id'])}</b> &nbsp; {esc(result['created_at'])}<br>
            Dataset: <b>{esc(result['dataset_label'])}</b><br>
            <span class="fs-mode">{esc(mode_text)}</span><br>
            <span class="fs-muted">Rule-based indicator, not a probability of fraud.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analysed", result["n_transactions"])
    c2.metric("Tools executed", len(result["executed_tools"]))
    c3.metric("LLM requests", result["llm_calls"])
    c4.metric("Limitations noted", len(result["limitations"]))

    st.subheader("Explanation")
    if result.get("llm_explanation"):
        exp = result["llm_explanation"]
        st.caption(f"Written by the LLM ({result['model']}) from the Python findings below.")
        st.write(llm_text(exp["summary"]))
        for obs in exp["key_observations"]:
            ids = f"  \nEvidence: {', '.join(obs['evidence_ids'])}" if obs["evidence_ids"] else ""
            st.markdown(f"- {llm_text(obs['statement'])}{ids}")
        if exp["caveats"]:
            st.markdown("**Caveats noted by the LLM**")
            for c in exp["caveats"]:
                st.markdown(f"- {llm_text(c)}")
    elif result.get("rule_based_summary"):
        st.caption("Rule-based template summary. No LLM was used.")
        st.write(result["rule_based_summary"])
    elif result.get("llm_explanation_error"):
        st.warning(result["llm_explanation_error"])

    st.subheader("How the score was calculated")
    contributions = pd.DataFrame(risk["contributions"])
    if contributions.empty:
        st.write("No indicators were found, so no points were added.")
    else:
        st.table(
            contributions[["label", "weight", "points", "note"]]
            .rename(columns={"label": "Indicator", "weight": "Weight", "points": "Points added", "note": "Note"})
            .set_index("Indicator")
        )
        if risk["uncapped_score"] > 100:
            st.caption(f"Raw total {risk['uncapped_score']} was capped at 100.")

    st.subheader("Findings and evidence")
    st.caption("Produced by deterministic Python tools.")
    for tool in result["tool_results"]:
        reason = result["tool_reasons"].get(tool["tool_name"], "")
        with st.expander(f"{tool['tool_name']}: {len(tool['findings'])} finding(s), status {tool['status']}", expanded=bool(tool["findings"])):
            if reason:
                st.markdown(f"Why it ran: {llm_text(reason)}")
            st.write(tool["explanation"])
            for f in tool["findings"]:
                label = INDICATOR_LABELS.get(f["indicator"], f["indicator"])
                st.markdown(f"**{label}** (account {f['account_id']})  \n{f['explanation']}")
                st.caption("Evidence IDs: " + ", ".join(f["evidence_ids"]))
            for lim in tool["limitations"]:
                st.caption(f"Limitation: {lim}")
    if result.get("rejected_tools"):
        st.warning(f"The LLM asked for tool(s) not on the allowlist, which were rejected: {', '.join(result['rejected_tools'])}")

    left, right = st.columns(2)
    with left:
        st.subheader("Missing evidence and limitations")
        if result["limitations"]:
            for lim in result["limitations"]:
                st.markdown(f"- {lim}")
        else:
            st.write("None recorded for the checks that ran.")
        st.markdown("- The data is synthetic and the thresholds are illustrative.")
    with right:
        st.subheader("Suggested next steps")
        for step in result["next_steps"]:
            st.markdown(f"- {step}")

    st.subheader("Agent execution trace")
    st.caption("Every node and tool that actually ran, in order. Actor shows whether the LLM, a Python tool or a fixed rule did the work.")
    st.table(trace_table(result["trace"]))

    st.subheader("Download")
    d1, d2 = st.columns(2)
    d1.download_button(
        "Download report (Markdown)",
        build_markdown_report(result),
        file_name=f"{result['investigation_id']}_report.md",
        mime="text/markdown",
        width="stretch",
    )
    d2.download_button(
        "Download findings (CSV)",
        build_findings_csv(result),
        file_name=f"{result['investigation_id']}_findings.csv",
        mime="text/csv",
        width="stretch",
    )


with tab_investigate:
    st.subheader("1. Choose the data")
    choice = st.selectbox("Dataset", list(BUILT_IN_DATASETS), key="dataset_choice")
    txn_file, login_file = BUILT_IN_DATASETS[choice]
    txn_bytes: bytes = (DATA_DIR / txn_file).read_bytes()
    login_bytes: bytes | None = (DATA_DIR / login_file).read_bytes() if login_file else None
    dataset_label = choice
    st.caption(f"Files: data/{txn_file}" + (f" and data/{login_file}" if login_file else " (no login-event file)"))

    txn_check: ValidationResult | None = None
    login_check: ValidationResult | None = None
    if txn_bytes is not None:
        txn_check = load_and_validate(txn_bytes, "transactions")
        show_validation(txn_check, "Transactions")
        if txn_check.is_valid:
            with st.expander("Preview transactions", expanded=False):
                preview(txn_check.data)
    if login_bytes is not None:
        login_check = load_and_validate(login_bytes, "logins")
        show_validation(login_check, "Login events")
        if login_check.is_valid:
            with st.expander("Preview login events", expanded=False):
                preview(login_check.data)

    st.subheader("2. Run the agent")
    data_ok = txn_check is not None and txn_check.is_valid and (login_check is None or login_check.is_valid)
    llm_ready = mode == MODE_RULE_BASED or bool(api_key)
    if not data_ok:
        st.info("Fix the validation errors above to enable the investigation.")
    if not llm_ready:
        st.warning("LLM Agent Mode needs a Groq API key. Add one, or choose Rule-Based Demo Mode in the sidebar.")

    current_key = fingerprint(txn_bytes or b"", login_bytes or b"", cfg.to_dict(), mode, model_name if mode == MODE_LLM else "")
    clicked = st.button("Investigate Transactions", type="primary", disabled=not (data_ok and llm_ready))

    if clicked:
        previous = st.session_state.get("last_result")
        reusable = (
            previous is not None
            and st.session_state.get("last_key") == current_key
            and previous["status"] == "completed"
            and not previous.get("llm_explanation_error")
        )
        if reusable:
            st.info("Same data and mode as the last run, so the previous result is shown and no new LLM request was made.")
        else:
            llm = None
            if mode == MODE_LLM:
                try:
                    llm = GroqService(api_key=api_key, model=model_name)
                except LLMServiceError as exc:
                    st.error(exc.message)
            with st.spinner("The agent is selecting and running checks..."):
                result = run_investigation(
                    txn_check.data,
                    login_check.data if login_check is not None else None,
                    dataset_label,
                    mode,
                    cfg,
                    llm,
                )
            st.session_state["last_result"] = result
            st.session_state["last_key"] = current_key
            try:
                save_investigation(result)
                st.session_state.pop("history_error", None)
            except (sqlite3.Error, OSError) as exc:
                st.session_state["history_error"] = f"History could not be saved: {exc}"

    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()
        if st.session_state.get("last_key") != current_key:
            st.caption("The data or mode changed after this result. Click Investigate Transactions to update it.")
        if st.session_state.get("history_error"):
            st.warning(st.session_state["history_error"])
        render_result(result)

# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

with tab_history:
    st.subheader("Previous investigations")
    
    try:
        records = list_investigations()
    except (sqlite3.Error, OSError) as exc:
        records = None
        st.warning(f"History is unavailable right now: {exc}. Investigations still work.")
    if records is not None:
        if not records:
            st.write("No investigations yet. Run one from the Investigate tab and it will appear here.")
        else:
            table = pd.DataFrame(records).rename(
                columns={
                    "investigation_id": "ID",
                    "created_at": "Date and time",
                    "dataset_label": "Dataset",
                    "n_transactions": "Transactions",
                    "risk_score": "Score",
                    "risk_category": "Category",
                    "findings_summary": "Findings",
                    "mode": "Mode",
                    "model": "Model",
                    "status": "Status",
                }
            )
            st.dataframe(table, hide_index=True, width="stretch")
            h1, h2 = st.columns(2)
            with h1:
                to_delete = st.selectbox("Delete one record", [r["investigation_id"] for r in records], key="delete_choice")
                if st.button("Delete selected record"):
                    try:
                        delete_investigation(to_delete)
                        st.rerun()
                    except (sqlite3.Error, OSError) as exc:
                        st.error(f"Could not delete: {exc}")
            with h2:
                confirm = st.checkbox("I want to delete all history", key="confirm_clear")
                if st.button("Delete all history", disabled=not confirm):
                    try:
                        clear_history()
                        st.rerun()
                    except (sqlite3.Error, OSError) as exc:
                        st.error(f"Could not clear history: {exc}")

# ---------------------------------------------------------------------------
# How it works tab
# ---------------------------------------------------------------------------

with tab_how:
    st.subheader("Agent workflow (LangGraph)")
    st.graphviz_chart(
        """
        digraph {
          rankdir=TB; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#F5EBDD", color="#D8C3A5", fontcolor="#4B2E24", fontname="Helvetica"];
          edge [color="#79665A", fontcolor="#79665A", fontname="Helvetica", fontsize=10];
          start [label="START", shape=oval, fillcolor="#FFFCF5"];
          v [label="validate_input"];
          s [label="summarize_dataset"];
          t [label="select_tools\\n(LLM call 1, or fixed rules)", fillcolor="#F6E1E2", color="#9B1C31"];
          x [label="execute_tools\\n(allowlisted Python tools)"];
          r [label="review_evidence"];
          a [label="score_and_aggregate\\n(deterministic score)"];
          e [label="explain_with_llm\\n(LLM call 2)", fillcolor="#F6E1E2", color="#9B1C31"];
          b [label="rule_based_summary"];
          f [label="finalize"];
          end [label="END", shape=oval, fillcolor="#FFFCF5"];
          start -> v; v -> s [label="valid"]; v -> f [label="invalid"];
          s -> t; t -> x [label="valid tools"]; t -> f [label="LLM error"];
          x -> r; r -> x [label="follow-up needed (max 2 rounds)"]; r -> a [label="enough evidence"];
          a -> e [label="LLM mode"]; a -> b [label="demo mode"];
          e -> f; b -> f; f -> end;
        }
        """
    )
    st.markdown(
        "- **The LLM decides which checks to run.** It receives a numeric summary of the data and returns tool names "
        "from a fixed allowlist. Python validates the choice before anything runs.\n"
        "- **Python does the analysis.** The tools and the risk score are deterministic; the LLM cannot change the score.\n"
        "- **The LLM explains the evidence.** Its second request turns the Python findings into a short explanation. "
        "Evidence IDs it cites are checked against the real findings.\n"
        "- **At most two LLM requests per investigation**, with no automatic retries."
    )

    st.subheader("Tools the agent can use")
    st.table(pd.DataFrame([{"Tool": k, "What it checks": v} for k, v in TOOL_DESCRIPTIONS.items()]))

    st.subheader("Risk score rules")
    st.table(
        pd.DataFrame(
            [
                {"Indicator": INDICATOR_LABELS[k], "Points": v, "Overlap group": INDICATOR_GROUPS[k]}
                for k, v in INDICATOR_WEIGHTS.items()
            ]
            + [{"Indicator": "Indicators from 2 or more different tools", "Points": MULTI_SIGNAL_BONUS, "Overlap group": "bonus"}]
        )
    )
    st.markdown(
        "Each indicator counts once. Within an overlap group only the highest-weighted indicator counts. "
        "The total is capped at 100. Bands: "
        + ", ".join(f"{low}-{high} {name}" for low, high, name in RISK_BANDS)
        + ". These weights are illustrative and not statistically validated."
    )

    st.subheader("Data format")
    st.markdown(
        f"**Transactions (required).** Columns: `transaction_id` (unique text), `account_id` (text), "
        f"`timestamp` (YYYY-MM-DD HH:MM:SS), `amount` (number above 0, INR), "
        f"`transaction_type` (one of {', '.join(sorted(ALLOWED_TRANSACTION_TYPES))}), `merchant` (text).\n\n"
        f"**Login events (optional).** Columns: `event_id` (unique text), `account_id`, `timestamp`, "
        f"`status` ({' or '.join(sorted(ALLOWED_LOGIN_STATUSES))}); `location` is optional.\n\n"
        f"Each file is limited to {MAX_ROWS:,} rows."
    )
