"""
Smoke tests for the Streamlit app using Streamlit's built-in AppTest.
These run the real app script headlessly (no browser, no LLM calls).
They check behaviour, not visual appearance.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("VERIFAI_DB_PATH", str(tmp_path / "app_history.db"))
    # Placeholder value = "no key" (also stops a local .env from supplying one).
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    return AppTest.from_file(str(APP_PATH), default_timeout=60).run()


def investigate_button(at):
    return next(b for b in at.button if b.label == "Investigate Transactions")


def test_app_loads_and_blocks_llm_mode_without_key(app):
    assert not app.exception
    assert investigate_button(app).disabled
    assert any("No GROQ_API_KEY found" in w.value for w in app.sidebar.warning)


def test_rule_based_investigation_runs_and_saves_history(app):
    app.sidebar.radio(key="mode_label").set_value("Rule-Based Demo Mode (no LLM)").run()
    app.selectbox(key="dataset_choice").set_value("Several suspicious indicators").run()
    investigate_button(app).click().run()
    assert not app.exception
    page = " ".join(m.value for m in app.markdown)
    assert "100<small> / 100</small>" in page
    assert "Risk category: Critical" in page
    assert any("No LLM was used" in c.value for c in app.caption)
    history = app.dataframe[-1].value
    assert len(history) == 1
    assert history.iloc[0]["Score"] == 100


def test_second_click_reuses_result_without_rerunning(app):
    app.sidebar.radio(key="mode_label").set_value("Rule-Based Demo Mode (no LLM)").run()
    investigate_button(app).click().run()
    first_id = app.session_state["last_result"]["investigation_id"]
    investigate_button(app).click().run()
    assert app.session_state["last_result"]["investigation_id"] == first_id
    assert any("no new LLM request was made" in i.value for i in app.info)


# ---------------------------------------------------------------------------
# LLM mode through the real app + real Groq SDK, against a local mock server.
# GROQ_BASE_URL is a standard Groq SDK setting; no real API call is made.
# ---------------------------------------------------------------------------

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


def start_mock_groq(status: int = 200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if status != 200:
                payload = {"error": {"message": "mock error"}}
            elif body["response_format"]["json_schema"]["name"] == "tool_selection":
                content = {"tools": [{"name": "unusual_amount_detector", "reason": "Amounts vary widely."}]}
                payload = None
            else:
                content = {
                    "summary": "MOCK LLM SUMMARY",
                    "key_observations": [{"statement": "Large transfer.", "evidence_ids": ["TXN-S007", "TXN-FAKE"]}],
                    "recommended_actions": ["Verify with the account holder."],
                    "caveats": ["Synthetic data."],
                }
                payload = None
            if payload is None:
                payload = {
                    "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(content)}, "finish_reason": "stop"}],
                }
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, calls


@pytest.fixture
def llm_app(monkeypatch, tmp_path):
    def make(status=200):
        server, calls = start_mock_groq(status)
        monkeypatch.setenv("VERIFAI_DB_PATH", str(tmp_path / "llm_history.db"))
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        monkeypatch.setenv("GROQ_BASE_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        return at, calls, server

    servers = []

    def factory(status=200):
        at, calls, server = make(status)
        servers.append(server)
        return at, calls

    yield factory
    for s in servers:
        s.shutdown()


def test_llm_mode_end_to_end_in_app(llm_app):
    at, calls = llm_app()
    assert not investigate_button(at).disabled
    assert calls == []  # opening the app makes no LLM request
    at.selectbox(key="dataset_choice").set_value("Several suspicious indicators").run()
    assert calls == []  # previewing data makes no LLM request
    investigate_button(at).click().run()
    assert not at.exception
    assert len(calls) == 2  # tool selection + explanation, nothing more
    result = at.session_state["last_result"]
    assert result["mode"] == "llm" and result["model"] == "openai/gpt-oss-20b"
    assert result["selected_tools"] == ["unusual_amount_detector"]
    assert result["llm_explanation"]["summary"] == "MOCK LLM SUMMARY"
    assert result["llm_explanation"]["key_observations"][0]["evidence_ids"] == ["TXN-S007"]  # invented ID removed
    assert any("Written by the LLM (openai/gpt-oss-20b)" in c.value for c in at.caption)


def test_llm_rate_limit_shows_error_and_no_fallback(llm_app):
    at, calls = llm_app(status=429)
    investigate_button(at).click().run()
    assert not at.exception
    assert len(calls) == 1  # stopped after the first failed request, no retries
    assert any("rate limit" in e.value for e in at.error)
    assert any("No rule-based result was substituted" in i.value for i in at.info)
    assert at.session_state["last_result"]["status"] == "failed"
