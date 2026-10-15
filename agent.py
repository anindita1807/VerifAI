"""
agent.py - The FraudSleuth AI agent, built with LangGraph.

Graph (nodes are functions, arrows are edges; [?] marks a conditional edge):

  START
    -> validate_input       [?] invalid data or missing LLM -> finalize
    -> summarize_dataset
    -> select_tools         (LLM call 1 in LLM mode; fixed rules in demo mode)
                            [?] no valid tools / LLM error -> finalize
    -> execute_tools        (runs only allowlisted Python tools)
    -> review_evidence      [?] follow-up check needed -> execute_tools (max 2 rounds)
    -> score_and_aggregate  (deterministic risk score)
                            [?] LLM mode -> explain_with_llm (LLM call 2)
                            [?] demo mode -> rule_based_summary
    -> finalize -> END

At most two LLM requests are made per investigation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, ValidationError

from groq_service import LLMClient, LLMServiceError
from tools import (
    INDICATOR_LABELS,
    REQUIRED_TRANSACTION_COLUMNS,
    AnalysisConfig,
    ToolResult,
    build_next_steps,
    calculate_risk_score,
    login_activity_checker,
    summarize_dataset,
    transaction_frequency_detector,
    unusual_amount_detector,
)

MODE_LLM = "llm"
MODE_RULE_BASED = "rule_based"
MAX_LLM_CALLS = 2
MAX_FOLLOWUP_ROUNDS = 2  # the review -> execute loop can repeat at most this many times
MAX_FINDINGS_SENT_TO_LLM = 15
MAX_IDS_PER_FINDING_SENT_TO_LLM = 8

# ---------------------------------------------------------------------------
# Tool allowlist: the ONLY functions the agent can run.
# ---------------------------------------------------------------------------

ToolFunction = Callable[[pd.DataFrame, pd.DataFrame | None, AnalysisConfig], ToolResult]

ALLOWED_TOOLS: dict[str, ToolFunction] = {
    "unusual_amount_detector": lambda txns, logins, cfg: unusual_amount_detector(txns, cfg),
    "transaction_frequency_detector": lambda txns, logins, cfg: transaction_frequency_detector(txns, cfg),
    "login_activity_checker": lambda txns, logins, cfg: login_activity_checker(txns, logins, cfg),
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "unusual_amount_detector": (
        "Flags transactions far above the account's own median amount (needs 5+ transactions per account) "
        "or above a fixed high-amount threshold."
    ),
    "transaction_frequency_detector": (
        "Flags many transactions in a short time window and repeated same-amount, same-merchant transactions."
    ),
    "login_activity_checker": (
        "Uses login-event data, if supplied, to flag repeated failed logins and transactions made soon after them."
    ),
}


class ToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    reason: str


class ToolSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[ToolChoice]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str
    evidence_ids: list[str]


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    key_observations: list[Observation]
    recommended_actions: list[str]
    caveats: list[str]


class AgentState(TypedDict, total=False):
    investigation_id: str
    created_at: str
    dataset_label: str
    mode: str
    model: str | None
    transactions: pd.DataFrame
    logins: pd.DataFrame | None
    config: AnalysisConfig
    dataset_summary: dict[str, Any]
    selected_tools: list[str]
    tool_reasons: dict[str, str]
    rejected_tools: list[str]
    executed_tools: list[str]
    tool_results: list[ToolResult]
    followup_tools: list[str]
    followup_rounds: int
    risk: dict[str, Any]
    next_steps: list[str]
    limitations: list[str]
    llm_explanation: dict[str, Any] | None
    llm_explanation_error: str | None
    rule_based_summary: str | None
    llm_calls: int
    status: str
    error: str | None
    error_kind: str | None
    trace: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _trace(state: AgentState, node: str, actor: str, status: str, detail: str) -> list[dict[str, str]]:
    """Return the trace with one more step appended (state is never mutated in place)."""
    steps = list(state.get("trace", []))
    steps.append({"step": str(len(steps) + 1), "node": node, "actor": actor, "status": status, "detail": detail})
    return steps


def validate_tool_selection(raw: Any) -> tuple[list[str], dict[str, str], list[str]]:
    """
    Check the LLM's tool choices. Returns (accepted tools, reasons, rejected names).
    Unknown tools are rejected, duplicates removed, reasons shortened.
    """
    try:
        selection = ToolSelection.model_validate(raw)
    except ValidationError as exc:
        raise LLMServiceError("invalid_response", f"The model's tool selection did not match the expected format: {exc.error_count()} error(s).") from exc
    accepted: list[str] = []
    reasons: dict[str, str] = {}
    rejected: list[str] = []
    for choice in selection.tools:
        name = choice.name.strip()
        if name not in ALLOWED_TOOLS:
            rejected.append(name[:60])
            continue
        if name in accepted:
            continue
        accepted.append(name)
        reasons[name] = choice.reason.strip()[:300] or "No reason given."
    return accepted, reasons, rejected


def rule_based_tool_selection(summary: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Fixed selection used in Rule-Based Demo Mode (no LLM)."""
    tools = ["unusual_amount_detector", "transaction_frequency_detector"]
    reasons = {
        "unusual_amount_detector": "Fixed rule: amount checks always run in demo mode.",
        "transaction_frequency_detector": "Fixed rule: frequency checks always run in demo mode.",
    }
    if summary.get("login_data_available"):
        tools.append("login_activity_checker")
        reasons["login_activity_checker"] = "Fixed rule: login data was supplied."
    return tools, reasons


def _all_evidence_ids(results: list[ToolResult]) -> set[str]:
    return {eid for r in results for f in r.findings for eid in f.evidence_ids}


def build_llm_evidence(state: AgentState) -> dict[str, Any]:
    """Compact evidence package for LLM call 2 (numbers and IDs, no merchant text)."""
    findings = []
    for result in state["tool_results"]:
        for f in result.findings:
            details = {k: v for k, v in f.details.items() if k != "merchant"}
            findings.append(
                {
                    "tool": result.tool_name,
                    "indicator": f.indicator,
                    "indicator_label": INDICATOR_LABELS.get(f.indicator, f.indicator),
                    "account_id": f.account_id,
                    "evidence_ids": f.evidence_ids[:MAX_IDS_PER_FINDING_SENT_TO_LLM],
                    "details": details,
                }
            )
    risk = state["risk"]
    return {
        "dataset_summary": state["dataset_summary"],
        "tools_run": state.get("executed_tools", []),
        "risk_score": risk["score"],
        "risk_category": risk["category"],
        "score_breakdown": [{"indicator": c["indicator"], "points": c["points"]} for c in risk["contributions"]],
        "findings": findings[:MAX_FINDINGS_SENT_TO_LLM],
        "findings_not_shown": max(0, len(findings) - MAX_FINDINGS_SENT_TO_LLM),
        "limitations": state.get("limitations", []),
        "rule_based_next_steps": state.get("next_steps", []),
    }


def check_explanation(raw: Any, known_ids: set[str]) -> tuple[dict[str, Any], int]:
    """
    Validate LLM call 2. Evidence IDs the model cites must exist in the tool
    results; unknown IDs are removed. Returns (clean explanation, removed count).
    """
    try:
        explanation = Explanation.model_validate(raw)
    except ValidationError as exc:
        raise LLMServiceError("invalid_response", f"The model's explanation did not match the expected format: {exc.error_count()} error(s).") from exc
    removed = 0
    observations = []
    for obs in explanation.key_observations:
        valid_ids = [i for i in obs.evidence_ids if i in known_ids]
        removed += len(obs.evidence_ids) - len(valid_ids)
        if obs.evidence_ids and not valid_ids:
            continue  # every cited ID was invented -> drop the statement
        observations.append({"statement": obs.statement.strip(), "evidence_ids": valid_ids})
    clean = {
        "summary": explanation.summary.strip(),
        "key_observations": observations,
        "recommended_actions": [a.strip() for a in explanation.recommended_actions if a.strip()][:8],
        "caveats": [c.strip() for c in explanation.caveats if c.strip()][:8],
    }
    return clean, removed


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(llm: LLMClient | None = None):
    """Build and compile the LangGraph workflow. `llm` is None in demo mode."""

    def validate_input(state: AgentState) -> dict[str, Any]:
        txns = state.get("transactions")
        if txns is None or len(txns) == 0:
            msg = "No valid transactions were supplied."
            return {"status": "failed", "error": msg, "error_kind": "invalid_input", "trace": _trace(state, "validate_input", "Python", "failed", msg)}
        missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in txns.columns]
        if missing:
            msg = f"Missing column(s): {', '.join(missing)}."
            return {"status": "failed", "error": msg, "error_kind": "invalid_input", "trace": _trace(state, "validate_input", "Python", "failed", msg)}
        if state["mode"] == MODE_LLM and llm is None:
            msg = "LLM Agent Mode needs a Groq API key. Add GROQ_API_KEY or choose Rule-Based Demo Mode."
            return {"status": "failed", "error": msg, "error_kind": "missing_key", "trace": _trace(state, "validate_input", "Python", "failed", msg)}
        detail = f"{len(txns)} transaction(s) passed validation."
        return {"status": "running", "trace": _trace(state, "validate_input", "Python", "completed", detail)}

    def summarize(state: AgentState) -> dict[str, Any]:
        summary = summarize_dataset(state["transactions"], state.get("logins"))
        detail = (
            f"{summary['transaction_count']} transactions, {summary['account_count']} account(s), "
            f"login data {'available' if summary['login_data_available'] else 'not available'}."
        )
        return {"dataset_summary": summary, "trace": _trace(state, "summarize_dataset", "Python", "completed", detail)}

    def select_tools(state: AgentState) -> dict[str, Any]:
        if state["mode"] == MODE_RULE_BASED:
            tools, reasons = rule_based_tool_selection(state["dataset_summary"])
            detail = "Fixed rules selected: " + ", ".join(tools)
            return {
                "selected_tools": tools,
                "tool_reasons": reasons,
                "rejected_tools": [],
                "trace": _trace(state, "select_tools", "Rule-based (no LLM)", "completed", detail),
            }

        calls = state.get("llm_calls", 0) + 1
        try:
            raw = llm.select_tools(state["dataset_summary"], TOOL_DESCRIPTIONS)
            tools, reasons, rejected = validate_tool_selection(raw)
        except LLMServiceError as exc:
            return {
                "llm_calls": calls,
                "status": "failed",
                "error": exc.message,
                "error_kind": exc.kind,
                "trace": _trace(state, "select_tools", f"LLM ({llm.model})", "failed", exc.message),
            }
        if not tools:
            msg = "The model did not select any allowlisted tool, so the investigation stopped."
            return {
                "llm_calls": calls,
                "rejected_tools": rejected,
                "status": "failed",
                "error": msg,
                "error_kind": "invalid_response",
                "trace": _trace(state, "select_tools", f"LLM ({llm.model})", "failed", msg),
            }
        detail = "LLM selected: " + ", ".join(tools)
        if rejected:
            detail += f". Rejected (not on allowlist): {', '.join(rejected)}"
        return {
            "llm_calls": calls,
            "selected_tools": tools,
            "tool_reasons": reasons,
            "rejected_tools": rejected,
            "trace": _trace(state, "select_tools", f"LLM ({llm.model})", "completed", detail),
        }

    def execute_tools(state: AgentState) -> dict[str, Any]:
        executed = list(state.get("executed_tools", []))
        results = list(state.get("tool_results", []))
        trace = list(state.get("trace", []))
        to_run = [t for t in state.get("selected_tools", []) + state.get("followup_tools", []) if t not in executed]
        for name in to_run:
            if name not in ALLOWED_TOOLS:  # defence in depth; selection is already validated
                continue
            try:
                result = ALLOWED_TOOLS[name](state["transactions"], state.get("logins"), state["config"])
            except (ValueError, KeyError, TypeError) as exc:
                result = ToolResult(name, "error", [], f"The tool failed: {exc}", [f"{name} failed, so its checks are missing."])
            results.append(result)
            executed.append(name)
            trace.append(
                {
                    "step": str(len(trace) + 1),
                    "node": "execute_tools",
                    "actor": "Python tool",
                    "status": result.status,
                    "detail": f"{name}: {result.explanation}",
                }
            )
        return {"executed_tools": executed, "tool_results": results, "trace": trace}

    def review_evidence(state: AgentState) -> dict[str, Any]:
        """
        Bounded follow-up loop: read the tool results and, if they suggest a
        missing check would help, send the agent back to execute_tools.
        Runs at most MAX_FOLLOWUP_ROUNDS times.
        """
        rounds = state.get("followup_rounds", 0)
        if rounds >= MAX_FOLLOWUP_ROUNDS:
            detail = f"Follow-up limit ({MAX_FOLLOWUP_ROUNDS} rounds) reached; moving on."
            return {"followup_tools": [], "trace": _trace(state, "review_evidence", "Python", "completed", detail)}
        executed = set(state.get("executed_tools", []))
        indicators = {f.indicator for r in state.get("tool_results", []) for f in r.findings}
        followup: list[str] = []
        reasons = dict(state.get("tool_reasons", {}))
        if indicators and state["dataset_summary"]["login_data_available"] and "login_activity_checker" not in executed:
            followup.append("login_activity_checker")
            reasons["login_activity_checker"] = "Follow-up rule: transaction indicators were found and login data is available."
        if indicators & {"repeated_failed_logins", "transaction_after_failed_logins"}:
            for name in ["unusual_amount_detector", "transaction_frequency_detector"]:
                if name not in executed:
                    followup.append(name)
                    reasons[name] = "Follow-up rule: login indicators were found, so the related transactions are checked too."
        followup = list(dict.fromkeys(followup))
        if followup:
            rounds += 1
            detail = f"Follow-up round {rounds}: evidence suggests another check: " + ", ".join(followup)
        else:
            detail = "No follow-up check needed; evidence is sufficient for scoring."
        return {
            "followup_tools": followup,
            "followup_rounds": rounds,
            "tool_reasons": reasons,
            "trace": _trace(state, "review_evidence", "Python", "completed", detail),
        }

    def score_and_aggregate(state: AgentState) -> dict[str, Any]:
        results = state.get("tool_results", [])
        risk = calculate_risk_score(results)
        executed = set(state.get("executed_tools", []))
        limitations: list[str] = []
        for r in results:
            limitations.extend(r.limitations)
        login_available = state["dataset_summary"]["login_data_available"]
        for name in ALLOWED_TOOLS:
            if name in executed:
                continue
            if name == "login_activity_checker" and not login_available:
                limitations.append(
                    "Login evidence was unavailable (no login-event data was supplied), so login activity was not "
                    "checked. This is missing evidence, not evidence that login activity was normal."
                )
            else:
                limitations.append(f"{name} was not run, so its indicators were not checked.")
        next_steps = build_next_steps(results, state["dataset_summary"]["login_data_available"])
        detail = f"Rule-based score {risk['score']}/100 ({risk['category']}) from {len(risk['contributions'])} contribution(s)."
        return {
            "risk": risk,
            "limitations": list(dict.fromkeys(limitations)),
            "next_steps": next_steps,
            "trace": _trace(state, "score_and_aggregate", "Python", "completed", detail),
        }

    def explain_with_llm(state: AgentState) -> dict[str, Any]:
        calls = state.get("llm_calls", 0)
        if calls >= MAX_LLM_CALLS:
            msg = "LLM request limit reached; explanation skipped."
            return {"llm_explanation": None, "llm_explanation_error": msg, "trace": _trace(state, "explain_with_llm", f"LLM ({llm.model})", "skipped", msg)}
        calls += 1
        try:
            raw = llm.explain(build_llm_evidence(state))
            explanation, removed = check_explanation(raw, _all_evidence_ids(state["tool_results"]))
        except LLMServiceError as exc:
            msg = f"Explanation unavailable: {exc.message} The Python findings and score above are still valid."
            return {
                "llm_calls": calls,
                "llm_explanation": None,
                "llm_explanation_error": msg,
                "trace": _trace(state, "explain_with_llm", f"LLM ({llm.model})", "failed", exc.message),
            }
        detail = "LLM wrote an explanation of the Python findings."
        if removed:
            detail += f" {removed} evidence ID(s) not found in the tool results were removed."
            explanation["caveats"].append(f"{removed} evidence reference(s) from the model could not be verified and were removed.")
        return {
            "llm_calls": calls,
            "llm_explanation": explanation,
            "llm_explanation_error": None,
            "trace": _trace(state, "explain_with_llm", f"LLM ({llm.model})", "completed", detail),
        }

    def rule_based_summary(state: AgentState) -> dict[str, Any]:
        risk = state["risk"]
        labels = [c["label"] for c in risk["contributions"] if c["points"] > 0 and c["indicator"] != "multiple_independent_signals"]
        if labels:
            text = (
                f"The rule-based checks produced a score of {risk['score']}/100 ({risk['category']}). "
                f"Indicators counted: {'; '.join(labels)}. These are patterns for review, not proof of fraud."
            )
        else:
            text = f"The rule-based checks found no indicators (score {risk['score']}/100, {risk['category']})."
        if state.get("limitations"):
            text += f" {len(state['limitations'])} limitation(s) apply; see below."
        return {
            "rule_based_summary": text,
            "llm_explanation": None,
            "trace": _trace(state, "rule_based_summary", "Rule-based (no LLM)", "completed", "Template summary written. No LLM was used."),
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        status = "failed" if state.get("status") == "failed" else "completed"
        detail = "Investigation complete." if status == "completed" else f"Investigation stopped: {state.get('error')}"
        return {"status": status, "trace": _trace(state, "finalize", "Python", status, detail)}

    # Conditional edge functions
    def after_validation(state: AgentState) -> str:
        return "finalize" if state.get("status") == "failed" else "summarize_dataset"

    def after_selection(state: AgentState) -> str:
        return "finalize" if state.get("status") == "failed" else "execute_tools"

    def after_review(state: AgentState) -> str:
        pending = [t for t in state.get("followup_tools", []) if t not in state.get("executed_tools", [])]
        return "execute_tools" if pending else "score_and_aggregate"

    def after_scoring(state: AgentState) -> str:
        return "explain_with_llm" if state["mode"] == MODE_LLM else "rule_based_summary"

    graph = StateGraph(AgentState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("summarize_dataset", summarize)
    graph.add_node("select_tools", select_tools)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("review_evidence", review_evidence)
    graph.add_node("score_and_aggregate", score_and_aggregate)
    graph.add_node("explain_with_llm", explain_with_llm)
    graph.add_node("rule_based_summary", rule_based_summary)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges("validate_input", after_validation, ["summarize_dataset", "finalize"])
    graph.add_edge("summarize_dataset", "select_tools")
    graph.add_conditional_edges("select_tools", after_selection, ["execute_tools", "finalize"])
    graph.add_edge("execute_tools", "review_evidence")
    graph.add_conditional_edges("review_evidence", after_review, ["execute_tools", "score_and_aggregate"])
    graph.add_conditional_edges("score_and_aggregate", after_scoring, ["explain_with_llm", "rule_based_summary"])
    graph.add_edge("explain_with_llm", "finalize")
    graph.add_edge("rule_based_summary", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_investigation(
    transactions: pd.DataFrame,
    logins: pd.DataFrame | None,
    dataset_label: str,
    mode: str,
    config: AnalysisConfig | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Run the agent once and return a plain dictionary of results (no DataFrames)."""
    if mode not in (MODE_LLM, MODE_RULE_BASED):
        raise ValueError(f"Unknown mode: {mode}")
    initial: AgentState = {
        "investigation_id": "INV-" + uuid.uuid4().hex[:8].upper(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_label": dataset_label,
        "mode": mode,
        "model": llm.model if (llm is not None and mode == MODE_LLM) else None,
        "transactions": transactions,
        "logins": logins,
        "config": config or AnalysisConfig(),
        "llm_calls": 0,
        "followup_rounds": 0,
        "trace": [],
        "status": "running",
    }
    final = build_graph(llm if mode == MODE_LLM else None).invoke(initial, {"recursion_limit": 25})
    return {
        "investigation_id": final["investigation_id"],
        "created_at": final["created_at"],
        "dataset_label": final["dataset_label"],
        "mode": final["mode"],
        "model": final.get("model"),
        "status": final["status"],
        "error": final.get("error"),
        "error_kind": final.get("error_kind"),
        "n_transactions": int(len(transactions)) if transactions is not None else 0,
        "n_login_events": int(len(logins)) if logins is not None else 0,
        "config": final["config"].to_dict(),
        "dataset_summary": final.get("dataset_summary"),
        "selected_tools": final.get("selected_tools", []),
        "tool_reasons": final.get("tool_reasons", {}),
        "rejected_tools": final.get("rejected_tools", []),
        "executed_tools": final.get("executed_tools", []),
        "tool_results": [r.to_dict() for r in final.get("tool_results", [])],
        "risk": final.get("risk"),
        "limitations": final.get("limitations", []),
        "next_steps": final.get("next_steps", []),
        "llm_explanation": final.get("llm_explanation"),
        "llm_explanation_error": final.get("llm_explanation_error"),
        "rule_based_summary": final.get("rule_based_summary"),
        "llm_calls": final.get("llm_calls", 0),
        "trace": final.get("trace", []),
    }
