# 🛡️ VerifAI — Intelligent Financial Fraud Investigation

### An AI-powered investigation system combining intelligent agents, evidence-driven analysis, and rule-based fraud detection.

<p align="center">

  **[🚀 Launch Live Application](https://verifai-fraud-investigation.streamlit.app/)**

</p>

VerifAI is a financial fraud investigation application designed to analyze transaction activity, identify suspicious indicators, organize supporting evidence, and generate investigation reports. It combines a **Groq-powered LLM agent workflow** with deterministic Python analysis tools, alongside a separate rule-based mode for investigations without LLM calls.

The goal is to make transaction investigations more structured, explainable, and easier to review by presenting findings alongside the evidence that supports them.

> **Note:** VerifAI is a demonstration and educational project. Its outputs are intended to support investigation workflows, not to establish fraud conclusively or replace professional review.

---

## ✨ Key Features

- **Dual Operating Modes**
  - **LLM Agent Mode (Groq):** An LLM (`openai/gpt-oss-20b`) chooses which investigation tools to run from a fixed allowlist and explains the resulting evidence. At most two LLM requests are made per investigation.
  - **Rule-Based Demo Mode:** Fixed Python rules choose the tools and a template writes the summary. No LLM calls are made.

- **Built-In Investigation Scenarios**
  - Normal transactions
  - Several suspicious indicators
  - Ambiguous activity with limited evidence

- **Evidence-Driven Analysis**
  - Examines transaction and available login-event data.
  - Presents every finding with the transaction or login IDs that support it.
  - Reports missing evidence (for example, no login data) as missing, never as normal activity.

- **Transparent Risk Assessment**
  - Produces a 0–100 rule-based risk score calculated by Python, with a table showing how each point was added.
  - The LLM cannot change the score.

- **Agent Execution Trace**
  - Shows every step that ran and whether the LLM, a Python tool or a fixed rule performed it.

- **Investigation Reports**
  - Downloadable Markdown reports.
  - Downloadable CSV of findings.

- **Investigation History**
  - Uses SQLite to store a summary of each investigation (never the dataset itself).

- **Interactive Web Interface**
  - Built with Streamlit for browser-based access.
  - Includes a live deployment so users can explore the application without setting up a local Python environment.

---

## 🚀 Live Demo

**Try VerifAI here:**

### [Open VerifAI — Live Application](https://verifai-fraud-investigation.streamlit.app/)

Explore the built-in scenarios in both operating modes and compare the resulting findings and evidence.

---

## 🧠 How It Works

VerifAI separates investigation coordination from evidence computation.

1. **Select a scenario**  
   Choose one of the built-in datasets representing normal activity, suspicious indicators, or ambiguous activity. Each file is validated before use.

2. **Choose an operating mode**  
   Run the investigation through the Groq-powered LLM agent or use the rule-based demo mode.

3. **Select tools**  
   In LLM Agent Mode, the LLM receives a numbers-only summary of the data (merchant names are never sent) and chooses tools from an allowlist. Python validates the choice and rejects any tool not on the list. In Rule-Based Demo Mode, fixed rules make this choice.

4. **Run the analysis**  
   Python tools check for unusually large amounts, bursts or repeats of transactions, and failed logins followed by transactions. A review step can add a missing check (at most two follow-up rounds).

5. **Score and explain**  
   Python calculates the risk score. In LLM Agent Mode, the LLM then explains the findings, and any evidence ID it cites is checked against the real findings.

6. **Review, export and retain results**  
   Examine the findings, risk score, evidence IDs and execution trace, download the reports, and review saved investigation history. The results should be reviewed in context rather than treated as definitive proof of fraud.

### High-Level Architecture

```text
                 ┌──────────────────────────┐
                 │      Streamlit UI        │
                 │ Scenario & Mode Selection│
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │ LangGraph Investigation  │
                 │          Flow            │
                 └────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
       ┌────────▼────────┐        ┌─────────▼────────┐
       │  LLM Agent Mode │        │ Rule-Based Mode  │
       │   Groq + Agent  │        │ Python Analysis  │
       └────────┬────────┘        └─────────┬────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │  Findings & Risk Score   │
                 │ Evidence & Investigation │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │ Reports & SQLite History │
                 └──────────────────────────┘
```

*Conceptual architecture illustrating the main application flow. The detailed LangGraph workflow is shown in the app's "How it works" tab.*

### Risk Score

| Indicator | Points |
|---|---|
| Amount far above the account's usual spending | 25 |
| Amount above the fixed review threshold (INR 1,00,000) | 15 |
| Many transactions in a short time | 20 |
| Repeated identical transactions | 15 |
| Repeated failed logins | 20 |
| Transaction soon after failed logins | 25 |
| Indicators from two or more different tools (bonus) | 10 |

Each indicator counts once, overlapping indicators count only once, and the total is capped at 100. Bands: **0–29 Low, 30–59 Medium, 60–79 High, 80–100 Critical.** These weights are illustrative and not statistically validated. All amounts are in Indian Rupees (INR).

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core application and analysis logic |
| Streamlit | Interactive web interface |
| Groq API (`openai/gpt-oss-20b`) | Language-model access for LLM Agent Mode |
| LangGraph | Agent workflow orchestration |
| Pandas | Tabular data processing |
| Pydantic | Validation of LLM output |
| SQLite | Investigation history storage |
| CSV | Built-in datasets and report export |
| pytest | Automated testing |
| GitHub | Source-code hosting and version control |
| Streamlit Community Cloud | Application deployment |

---

## 📂 Project Structure

```text
VerifAI/
├── app.py                  # Streamlit application
├── agent.py                # LangGraph agent workflow
├── groq_service.py         # Groq API integration
├── tools.py                # Validation, investigation tools and risk scoring
├── database.py             # SQLite history management
├── report_generator.py     # Markdown and CSV report generation
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test configuration
├── .env.example            # Environment variable template
├── .streamlit/
│   └── config.toml         # Theme settings
├── assets/                 # Application logo
├── data/                   # Built-in synthetic datasets
└── tests/                  # Automated tests
```

---

## ⚙️ Run Locally

### Prerequisites

- Python 3.12 (the version the project was tested with).
- Git, if you want to clone the repository.
- A Groq API key for LLM Agent Mode.

### 1. Clone the repository

```bash
git clone https://github.com/anindita1807/VerifAI.git
cd VerifAI
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Copy `.env.example` to a new file named `.env` and replace the placeholder with your key:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

`GROQ_MODEL` is optional; the default is `openai/gpt-oss-20b`. Keep `.env` out of version control. Never commit API keys to GitHub.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

**Rule-Based Demo Mode** can be used without a key or LLM calls. LLM Agent Mode requires a valid Groq API key.

---

## 🧪 Run the Tests

```bash
python -m pytest -v
```

The tests cover data validation, the three investigation tools, risk-score boundaries, rejection of tools that are not on the allowlist, the follow-up loop, LLM error handling, SQLite storage, report generation and the Streamlit app. They never call the real Groq API (the LLM is replaced by mocks), so no API key is needed and no quota is used.

---

## ☁️ Deployment

VerifAI is deployed using **Streamlit Community Cloud** and its source code is hosted on GitHub.

**Live application:** [https://verifai-fraud-investigation.streamlit.app/](https://verifai-fraud-investigation.streamlit.app/)

For your own deployment:

1. Push the project files to a GitHub repository.
2. Connect the repository to Streamlit Community Cloud.
3. Set the main file path to `app.py` and choose Python 3.12 in the advanced settings.
4. Add required secrets in the app's Streamlit settings.

For Groq integration, the secrets are configured in TOML format:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-20b"   # optional
```

Replace the example value with your own key. Do not put real secrets in this README or in source code.

---

## 🔐 Security and Responsible Use

- **Protect API credentials:** Store secrets in Streamlit Secrets or a local `.env` file, never in code.
- **Minimal data to the LLM:** Only a numeric summary and the findings (IDs and numbers) are sent to Groq; merchant names are never sent.
- **Use synthetic demonstration data:** The built-in scenarios are intended for exploring application functionality, not for validating the system against real-world fraud cases.
- **Review evidence carefully:** Suspicious indicators are signals for further investigation, not proof of wrongdoing.
- **Keep human oversight:** The application takes no action on accounts. Financial decisions should involve appropriate human review and established organizational controls.
- **Understand deployment persistence:** The SQLite history on Streamlit Community Cloud is temporary and can be cleared when the app restarts or is redeployed.

---

## 🎯 Project Objectives

VerifAI demonstrates how AI-agent orchestration can be combined with conventional Python analysis in a financial investigation workflow.

Its main objectives are to:

- Explore the use of LLM agents for coordinating investigation steps.
- Keep analytical computations grounded in tool-generated evidence.
- Provide a rule-based alternative for comparison and demonstrations.
- Present risk findings and evidence in a reviewable format.
- Generate structured reports to support investigation documentation.

---

## 🔮 Future Enhancements

Potential directions for future development include:

- Secure ingestion of approved transaction datasets.
- Expanded fraud-pattern and anomaly-detection techniques.
- Additional evidence correlation and investigation workflows.
- Improved explainability and audit trails.
- Evaluation against labeled datasets to measure accuracy.
- Persistent cloud storage and access controls for multi-user deployments.

These are potential enhancements, not features currently claimed as implemented.

---

## 👩‍💻 Author

**Anindita Bandopadhyaya**

Developer of VerifAI — Intelligent Financial Fraud Investigation.

- **GitHub:** [@anindita1807](https://github.com/anindita1807)
- **Live Project:** [Launch VerifAI](https://verifai-fraud-investigation.streamlit.app/)

---


</p>
