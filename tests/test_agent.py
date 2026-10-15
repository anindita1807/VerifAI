"""
Tests for the LangGraph workflow and the Groq service.

No real API calls are made:
  - FakeLLM stands in for the LLM in workflow tests.
  - GroqService tests use the real Groq SDK with a fake HTTP transport, so the
    request format and error handling are exercised without a key or quota.
"""

import json

import groq
import httpx
import pytest

from agent import MODE_LLM, MODE_RULE_BASED, check_explanation, run_investigation, validate_tool_selection
from groq_service import GroqService, LLMServiceError


class FakeLLM:
    """Records calls and returns scripted responses (or raises scripted errors)."""

    model = "fake-model"

    def __init__(self, selection=None, explanation=None):
        self.selection = selection
        self.explanation = explanation
        self.calls = []

    def _answer(self, value):
        if isinstance(value, Exception):
            raise value
        return value

    def select_tools(self, dataset_summary, tool_descriptions):
        self.calls.append(("select_tools", dataset_summary))
        return self._answer(self.selection)

    def explain(self, evidence):
        self.calls.append(("explain", evidence))
        return self._answer(self.explanation)


GOOD_EXPLANATION = {
    "summary": "Several indicators on ACC-S01 need review.",
    "key_observations": [{"statement": "A large transfer followed failed logins.", "evidence_ids": ["TXN-S007", "LGN-S008"]}],
    "recommended_actions": ["Verify the transfer with the account holder."],
    "caveats": ["Synthetic data."],
}


def selection(*names):
    return {"tools": [{"name": n, "reason": f"Reason for {n}"} for n in names]}


# ------------------------------------------------------------ workflow tests


def test_llm_selection_drives_tools_and_follow_up_is_bounded(suspicious_data):
    txns, logins = suspicious_data
    llm = FakeLLM(selection("unusual_amount_detector"), GOOD_EXPLANATION)
    result = run_investigation(txns, logins, "suspicious", MODE_LLM, llm=llm)

    assert result["status"] == "completed"
    assert result["selected_tools"] == ["unusual_amount_detector"]
    # Round 1: amount finding + login data -> login checker runs.
    # Round 2: login findings -> frequency detector runs. Then the loop stops.
    assert result["executed_tools"] == ["unusual_amount_detector", "login_activity_checker", "transaction_frequency_detector"]
    assert [c[0] for c in llm.calls] == ["select_tools", "explain"]
    assert result["llm_calls"] == 2
    assert result["risk"]["score"] == 100
    reviews = [t["detail"] for t in result["trace"] if t["node"] == "review_evidence"]
    assert reviews[0].startswith("Follow-up round 1") and reviews[1].startswith("Follow-up round 2")
    assert reviews[2].startswith("Follow-up limit")
    actors = {t["node"]: t["actor"] for t in result["trace"]}
    assert actors["select_tools"] == "LLM (fake-model)"
    assert actors["score_and_aggregate"] == "Python"


def test_llm_sees_summary_without_merchant_text(suspicious_data):
    txns, logins = suspicious_data
    llm = FakeLLM(selection("transaction_frequency_detector"), GOOD_EXPLANATION)
    run_investigation(txns, logins, "suspicious", MODE_LLM, llm=llm)
    sent = json.dumps([c[1] for c in llm.calls], default=str)
    assert "Gadget Hub Online" not in sent
    assert "Quickpay" not in sent


def test_tools_not_on_allowlist_are_rejected(normal_data):
    txns, logins = normal_data
    llm = FakeLLM(selection("delete_database", "unusual_amount_detector", "os.system", "unusual_amount_detector"), GOOD_EXPLANATION)
    result = run_investigation(txns, logins, "normal", MODE_LLM, llm=llm)
    assert result["rejected_tools"] == ["delete_database", "os.system"]
    assert result["executed_tools"] == ["unusual_amount_detector"]


def test_only_invalid_tools_stops_the_investigation(normal_data):
    txns, logins = normal_data
    result = run_investigation(txns, logins, "normal", MODE_LLM, llm=FakeLLM(selection("run_sql")))
    assert result["status"] == "failed"
    assert result["executed_tools"] == []
    assert result["risk"] is None


def test_malformed_tool_selection_is_rejected():
    with pytest.raises(LLMServiceError) as err:
        validate_tool_selection({"tools": [{"name": "unusual_amount_detector"}], "extra": True})
    assert err.value.kind == "invalid_response"


def test_rate_limit_stops_without_silent_rule_based_fallback(suspicious_data):
    txns, logins = suspicious_data
    llm = FakeLLM(LLMServiceError("rate_limit", "The Groq free-tier rate limit was reached."))
    result = run_investigation(txns, logins, "suspicious", MODE_LLM, llm=llm)
    assert result["status"] == "failed"
    assert result["error_kind"] == "rate_limit"
    assert result["tool_results"] == []
    assert result["rule_based_summary"] is None
    assert result["llm_calls"] == 1


def test_explanation_failure_keeps_python_findings(suspicious_data):
    txns, logins = suspicious_data
    llm = FakeLLM(selection("unusual_amount_detector", "transaction_frequency_detector", "login_activity_checker"),
                  LLMServiceError("timeout", "The request to Groq timed out."))
    result = run_investigation(txns, logins, "suspicious", MODE_LLM, llm=llm)
    assert result["status"] == "completed"
    assert result["risk"]["score"] == 100
    assert result["llm_explanation"] is None
    assert "timed out" in result["llm_explanation_error"]
    assert result["rule_based_summary"] is None  # no silent substitution


def test_invented_evidence_ids_are_removed():
    raw = {
        "summary": "s",
        "key_observations": [
            {"statement": "real", "evidence_ids": ["TXN-1", "TXN-FAKE"]},
            {"statement": "invented", "evidence_ids": ["TXN-999"]},
        ],
        "recommended_actions": [],
        "caveats": [],
    }
    clean, removed = check_explanation(raw, {"TXN-1"})
    assert removed == 2
    assert clean["key_observations"] == [{"statement": "real", "evidence_ids": ["TXN-1"]}]


def test_rule_based_mode_makes_no_llm_calls(ambiguous_data):
    txns, logins = ambiguous_data
    llm = FakeLLM(selection("unusual_amount_detector"), GOOD_EXPLANATION)
    result = run_investigation(txns, logins, "ambiguous", MODE_RULE_BASED, llm=llm)
    assert llm.calls == []
    assert result["llm_calls"] == 0
    assert result["model"] is None
    assert result["risk"]["score"] == 40 and result["risk"]["category"] == "Medium"
    assert any("Login evidence was unavailable" in lim for lim in result["limitations"])
    assert "No LLM was used" in result["trace"][-2]["detail"]


@pytest.mark.parametrize(
    "fixture,score,category",
    [("normal_data", 0, "Low"), ("suspicious_data", 100, "Critical"), ("ambiguous_data", 40, "Medium")],
)
def test_end_to_end_demo_datasets(request, fixture, score, category):
    txns, logins = request.getfixturevalue(fixture)
    result = run_investigation(txns, logins, fixture, MODE_RULE_BASED)
    assert result["status"] == "completed"
    assert (result["risk"]["score"], result["risk"]["category"]) == (score, category)
    assert result["trace"][-1]["node"] == "finalize"


def test_llm_mode_without_client_fails_clearly(normal_data):
    txns, logins = normal_data
    result = run_investigation(txns, logins, "normal", MODE_LLM, llm=None)
    assert result["status"] == "failed"
    assert result["error_kind"] == "missing_key"


# ------------------------------------------------------------ Groq service tests


def make_service(handler):
    """GroqService using the real SDK but a fake HTTP transport."""
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = groq.Groq(api_key="test-key", http_client=http_client, max_retries=0)
    return GroqService(api_key="test-key", client=client)


def chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def test_groq_request_format_and_successful_parse():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response(json.dumps(selection("unusual_amount_detector"))))

    service = make_service(handler)
    out = service.select_tools({"transaction_count": 3}, {"unusual_amount_detector": "d"})
    assert out == selection("unusual_amount_detector")
    body = requests[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["reasoning_effort"] == "low"
    assert body["include_reasoning"] is False
    assert body["temperature"] == 0


@pytest.mark.parametrize(
    "status,kind",
    [(401, "auth"), (429, "rate_limit"), (404, "model_unavailable"), (400, "bad_request"), (503, "api_error")],
)
def test_groq_http_errors_are_mapped_and_not_retried(status, kind):
    count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        return httpx.Response(status, json={"error": {"message": "test error"}}, headers={"retry-after": "7"})

    with pytest.raises(LLMServiceError) as err:
        make_service(handler).explain({"x": 1})
    assert err.value.kind == kind
    assert count["n"] == 1  # no automatic retries that waste free-tier quota
    if kind == "rate_limit":
        assert "7 seconds" in err.value.message


def test_groq_network_error_and_invalid_json():
    def network_down(request):
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(LLMServiceError) as err:
        make_service(network_down).explain({})
    assert err.value.kind == "network"

    def bad_json(request):
        return httpx.Response(200, json=chat_response("this is not json"))

    with pytest.raises(LLMServiceError) as err:
        make_service(bad_json).explain({})
    assert err.value.kind == "invalid_response"
