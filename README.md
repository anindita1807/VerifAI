# VerifAI

**An Agentic AI system for suspicious transaction analysis** (CA3 project, Agentic AI)

> This application is an educational transaction-analysis prototype. Its risk score is a rule-based indicator, not a probability of fraud. All findings require human review.

VerifAI analyses built-in synthetic transaction data and, where available, login events. An LLM agent built with **LangGraph** looks at a numeric summary of the data and **chooses which Python investigation tools to run** from a fixed allowlist. The tools find patterns such as unusually large amounts, bursts of transactions and failed logins. Python then calculates a transparent 0–100 risk score, and the LLM writes a short explanation of the evidence for a human reviewer.

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Technology stack](#2-technology-stack)
3. [Project structure](#3-project-structure)
4. [Run it on Windows (step by step)](#4-run-it-on-windows-step-by-step)
5. [Run the tests](#5-run-the-tests)
6. [Upload to GitHub](#6-upload-to-github)
7. [Deploy on Streamlit Community Cloud](#7-deploy-on-streamlit-community-cloud)
8. [How the agent works](#8-how-the-agent-works)
9. [Risk scoring rules](#9-risk-scoring-rules)
10. [CSV format](#10-csv-format)
11. [Built-in datasets](#11-built-in-datasets)
12. [Groq free tier and LLM usage](#12-groq-free-tier-and-llm-usage)
13. [Theme and colours](#13-theme-and-colours)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations and ethics](#15-limitations-and-ethics)

---

## 1. What the project does

- Choose one of three built-in synthetic datasets (uploading your own data is not supported).
- Preview and validate the data (column checks, timestamps, amounts, duplicates, row count).
- Click **Investigate Transactions** to run the agent:
  1. The LLM selects investigation tools from an allowlist (LLM request 1).
  2. Python validates the choice and runs the tools.
  3. A bounded follow-up loop adds a missing check if the evidence calls for it.
  4. Python calculates the rule-based risk score.
  5. The LLM explains the findings, citing evidence IDs (LLM request 2).
- See the score, indicators, evidence, limitations, next steps and the **agent execution trace**.
- Download a Markdown report or a CSV of findings.
- View or delete past investigations (SQLite).

Two operating modes, always clearly labelled:

| Mode | Who chooses tools | Who writes the explanation | LLM requests |
|---|---|---|---|
| **LLM Agent Mode (Groq)** | The LLM (validated by Python) | The LLM | At most 2 |
| **Rule-Based Demo Mode** | Fixed Python rules | A Python template | 0 |

If the LLM fails (bad key, rate limit, network), the app shows the error and **does not** silently switch modes. You can choose Rule-Based Demo Mode yourself.

## 2. Technology stack

| Part | Technology |
|---|---|
| Language | Python 3.12 |
| LLM | `openai/gpt-oss-20b` on the Groq API (free plan) |
| Agent framework | LangGraph 1.2 |
| Frontend | Streamlit 1.66 |
| Data processing | pandas 3.0 |
| Output validation | Pydantic 2 |
| Storage | CSV files (input) and SQLite (history) |
| Testing | pytest |
| Deployment | Streamlit Community Cloud |

Exact versions are in `requirements.txt`. The model ID, free-plan availability and rate limits were checked against Groq's official documentation on 14 October 2026.

## 3. Project structure

```
VerifAI/
├── app.py                  Streamlit user interface
├── agent.py                LangGraph agent: state, nodes, edges, tool allowlist
├── tools.py                Validation, the 3 investigation tools, risk scoring
├── groq_service.py         The only file that calls the Groq API
├── database.py             SQLite investigation history
├── report_generator.py     Markdown and CSV reports
├── requirements.txt        Python packages (exact tested versions)
├── pytest.ini              Test settings
├── .env.example            Template for your local API key
├── .gitignore              Keeps secrets and databases out of Git
├── assets/                 VerifAI logo (SVG and PNG)
├── .streamlit/
│   ├── config.toml         Theme colours
│   └── secrets.toml.example  Template for Streamlit Cloud secrets
├── data/                   Synthetic datasets (5 CSV files)
├── tests/                  Automated tests (pytest)
└── docs/                   CA3 report, rubric mapping, demo script, viva questions
```

## 4. Run it on Windows (step by step)

### 4.1 Install Python and Git (once)

1. Install **Python 3.12** from <https://www.python.org/downloads/>. On the first installer screen, tick **"Add python.exe to PATH"**.
2. Install **Git** from <https://git-scm.com/download/win> (default options are fine).
3. Install **VS Code** from <https://code.visualstudio.com/> and add the **Python** extension (by Microsoft).

Check in PowerShell:

```powershell
python --version
git --version
```

### 4.2 Open the project in VS Code

1. Unzip `VerifAI.zip`, for example to `C:\Projects\VerifAI`.
2. In VS Code: **File > Open Folder** and choose the `VerifAI` folder.
3. Open a terminal: **Terminal > New Terminal** (it opens in the project folder).

### 4.3 Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of the prompt. If PowerShell says *running scripts is disabled*, run this once and try again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

In VS Code, also pick this interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

### 4.4 Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.5 Add your Groq API key (for LLM Agent Mode)

1. Create a free account at <https://console.groq.com> and open **API Keys > Create API Key**. Copy the key.
2. Create your local `.env` file from the template:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Open `.env` in VS Code and replace `your_groq_api_key_here` with your key:

   ```
   GROQ_API_KEY=gsk_your_real_key
   GROQ_MODEL=openai/gpt-oss-20b
   ```

`.env` is listed in `.gitignore`, so Git will not upload it. Never put the key in a `.py` file.

Without a key the app still runs; you can use Rule-Based Demo Mode.

### 4.6 Run the app

```powershell
python -m streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Stop the app with **Ctrl+C** in the terminal.

## 5. Run the tests

```powershell
python -m pytest -v
```

The tests never call the real Groq API and need no key. LLM calls are replaced by a fake LLM, a fake HTTP transport, or a small local mock server, so they use no quota. They cover validation, all three tools, scoring boundaries and capping, tool-selection validation, the LangGraph workflow, rate-limit and API error handling, SQLite, reports, and the Streamlit app itself (using Streamlit's AppTest, which checks behaviour, not visual appearance).

Result when the project was built (Python 3.12, fresh virtual environment): **58 passed**. Run the command yourself and keep a screenshot of your own output as evidence.

## 6. Upload to GitHub

1. Create an **empty** repository on <https://github.com/new> (no README, no .gitignore), for example `VerifAI`.
2. In the VS Code terminal (inside the project folder):

   ```powershell
   git init
   git add .
   git status
   ```

   Check the `git status` list: `.env`, `.venv` and `*.db` must **not** appear. If `.env` appears, stop and check `.gitignore`.

3. Commit and push (replace `YOUR-USERNAME`):

   ```powershell
   git commit -m "VerifAI - CA3 Agentic AI project"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/VerifAI.git
   git push -u origin main
   ```

## 7. Deploy on Streamlit Community Cloud

1. Push the project to GitHub (section 6). A **public** repository is simplest.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** and choose **Deploy a public app from GitHub**, then fill in:
   - Repository: `YOUR-USERNAME/VerifAI`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a name, e.g. `verifai`
4. Open **Advanced settings**:
   - **Python version:** choose **3.12** (the version used for testing).
   - **Secrets:** paste the contents of `.streamlit/secrets.toml.example` with your real key:

     ```toml
     GROQ_API_KEY = "gsk_your_real_key"
     GROQ_MODEL = "openai/gpt-oss-20b"
     ```

5. Click **Deploy**. The first build takes a few minutes.
6. **Test the public URL:** open it in a private/incognito window, run the *Several suspicious indicators* dataset in LLM Agent Mode, and confirm that the trace shows `LLM (openai/gpt-oss-20b)` for `select_tools` and `explain_with_llm`.

To change secrets later: open your app, then **⋮ > Settings > Secrets**. The app reads `st.secrets["GROQ_API_KEY"]` on Streamlit Cloud and `.env` locally.

**Important:**
- The deployed app works only while Groq is reachable and your free-tier quota has capacity.
- **History is not durable on Streamlit Cloud.** The SQLite file lives on the app's temporary disk and can be wiped when the app restarts, goes to sleep or is redeployed. The app keeps working if history is lost.
- Free apps go to sleep after a period without visitors; the first visit then takes a moment to wake up.

## 8. How the agent works

```
START
  -> validate_input        invalid data / no key ------------------> finalize
  -> summarize_dataset     (numbers only; merchant names never sent to the LLM)
  -> select_tools          LLM request 1 (or fixed rules in demo mode)
                           LLM error / no valid tool -------------> finalize
  -> execute_tools         runs ONLY allowlisted Python functions
  -> review_evidence       follow-up needed? -> execute_tools (max 2 rounds)
  -> score_and_aggregate   deterministic score, limitations, next steps
  -> explain_with_llm      LLM request 2   (LLM mode)
     or rule_based_summary Python template (demo mode)
  -> finalize -> END
```

- **State:** a `TypedDict` (`AgentState` in `agent.py`) holding the data, selected tools, tool results, score, trace and counters.
- **Nodes:** plain Python functions, each returning an update to the state.
- **Edges:** fixed edges plus four conditional edges (the routing functions `after_validation`, `after_selection`, `after_review`, `after_scoring`).
- **Tool allowlist:** `ALLOWED_TOOLS` in `agent.py`. The LLM can only name tools; Python validates the names with Pydantic and rejects anything else. The LLM cannot run code, SQL or file operations.
- **Bounded:** at most 2 LLM requests, at most 2 follow-up rounds, and only 3 tools exist.
- **Evidence check:** evidence IDs cited by the LLM are compared with the real tool results; invented IDs are removed.

## 9. Risk scoring rules

The score is calculated only in Python (`calculate_risk_score` in `tools.py`). The LLM cannot change it.

| Indicator | Points | Overlap group |
|---|---|---|
| Amount far above the account's usual spending | 25 | amount |
| Amount above the fixed review threshold | 15 | amount |
| Many transactions in a short time | 20 | frequency |
| Repeated identical transactions | 15 | frequency |
| Repeated failed logins | 20 | login failures |
| Transaction soon after failed logins | 25 | login to transaction |
| Indicators from 2 or more different tools (bonus) | 10 | bonus |

1. Each indicator counts **once**, however many times it appears.
2. Within an overlap group only the **highest** indicator counts (no double counting).
3. Add the bonus if two or more different tools found something.
4. **Cap at 100.**

Bands: **0–29 Low, 30–59 Medium, 60–79 High, 80–100 Critical.** These weights and thresholds are illustrative choices for a demonstration, not industry standards, and they are not statistically validated. The thresholds are fixed defaults defined in `AnalysisConfig` in `tools.py`.

## 10. Data format

The built-in datasets in `data/` follow this format (uploading your own CSV is not supported in the app).

**Transactions**

| Column | Type | Rules |
|---|---|---|
| `transaction_id` | text | required, unique |
| `account_id` | text | required |
| `timestamp` | date-time | `YYYY-MM-DD HH:MM:SS` (ISO 8601) |
| `amount` | number | greater than 0 (INR) |
| `transaction_type` | text | one of `purchase`, `transfer`, `withdrawal`, `payment`, `deposit`, `refund` |
| `merchant` | text | required |

**Login events (optional, separate file)**

| Column | Type | Rules |
|---|---|---|
| `event_id` | text | required, unique |
| `account_id` | text | required |
| `timestamp` | date-time | `YYYY-MM-DD HH:MM:SS` |
| `status` | text | `success` or `failed` |
| `location` | text | optional |

Each file is validated before use. Invalid rows are reported with their CSV line numbers; they are never silently dropped.

## 11. Built-in datasets

| Dataset | Files | Rule-based result |
|---|---|---|
| Normal transactions | `normal_transactions.csv`, `normal_login_events.csv` | 0 / 100, Low |
| Several suspicious indicators | `suspicious_transactions.csv`, `suspicious_login_events.csv` | 100 / 100, Critical |
| Ambiguous activity, limited evidence | `ambiguous_transactions.csv` (no login file) | 40 / 100, Medium, with 3 limitations |

All data is invented. In LLM mode the score can differ from this table only if the LLM skips a tool; the trace and limitations always show which tools ran.

## 12. Groq free tier and LLM usage

Groq's documented Free Plan limits for `openai/gpt-oss-20b` (checked 14 October 2026): **30 requests/minute, 1,000 requests/day, 8,000 tokens/minute, 200,000 tokens/day**. Limits apply to your whole Groq organisation and can change; check <https://console.groq.com/docs/rate-limits>.

How the app saves quota:

- No LLM request when the app opens, or when you preview data.
- At most **2 requests per investigation**; **no automatic retries** (`max_retries=0`).
- Only a compact numeric summary and the findings (IDs and numbers) are sent, never the whole CSV.
- `reasoning_effort="low"`, a 1,200-token output cap and a 20-second timeout.
- Clicking **Investigate Transactions** again with the same data and mode reuses the previous result instead of calling the API.

If Groq retires the model, set `GROQ_MODEL` (in `.env` or Streamlit secrets) to a current model that supports JSON Schema mode; check <https://console.groq.com/docs/models>.

## 13. Theme and colours

The palette is set in `.streamlit/config.toml`, with a little extra CSS at the top of `app.py`.

| Role | Colour |
|---|---|
| Cherry red (primary accent, buttons) | `#9B1C31` |
| Deep cherry red (hover, strong accent) | `#741426` |
| Cream (sidebar, panels) | `#F5EBDD` |
| Ivory (page background) | `#FFFCF5` |
| Coffee brown (headings, text) | `#4B2E24` |
| Light coffee / tan (borders) | `#D8C3A5` |
| Muted brown (secondary text) | `#79665A` |

Fonts: Spectral (headings) and Work Sans (body), loaded from Google Fonts. The logo (shield, magnifying glass and checkmark) is in `assets/`. Risk categories are always written as text, not shown by colour alone.

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognised | Reinstall Python and tick "Add python.exe to PATH", then reopen VS Code. |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. |
| `ModuleNotFoundError` | Activate `.venv` and run `python -m pip install -r requirements.txt`. |
| Sidebar says "No GROQ_API_KEY found" | Check `.env` is in the project folder (not `.env.example`) and restart the app. On Streamlit Cloud, check **Settings > Secrets**. |
| "Groq rejected the API key" | Create a new key in the Groq console and update `.env` or secrets. |
| "rate limit was reached" | Wait a minute (or until the daily limit resets). Use Rule-Based Demo Mode meanwhile. |
| "model ... was not found" | The model was retired. Set `GROQ_MODEL` to a supported model. |
| "Could not reach Groq" | Check your internet connection or firewall. |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502` |
| Dataset fails validation | Read the listed line numbers and compare with section 10. |
| Streamlit Cloud build fails | Check the build logs; choose Python 3.12 in Advanced settings; confirm `requirements.txt` is at the repository root. |
| History disappeared on Streamlit Cloud | Expected: the SQLite file is temporary there (see section 7). |

## 15. Limitations and ethics

- Synthetic data only; the app has no upload option.
- The score is a rule-based indicator, not a fraud probability, and the thresholds are not validated.
- A pattern is not proof. Large amounts, bursts and logins can all have innocent explanations.
- Missing login data is reported as missing evidence, never as normal activity.
- The LLM can make mistakes; its text is labelled and its evidence IDs are checked.
- In LLM mode a numeric summary and the findings are sent to Groq (a third-party service).
- The tool never blocks accounts or takes any action; a human reviewer decides.
