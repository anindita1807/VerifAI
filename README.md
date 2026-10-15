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
  - **LLM Agent Mode (Groq):** Uses a language-model-powered agent to coordinate permitted investigation tools and explain the resulting evidence.
  - **Rule-Based Demo Mode:** Runs rule-based analysis without making LLM calls.

- **Built-In Investigation Scenarios**
  - Normal transactions
  - Several suspicious indicators
  - Ambiguous activity with limited evidence

- **Evidence-Driven Analysis**
  - Examines transaction and available login-event data.
  - Presents investigation findings with supporting evidence identifiers.
  - Helps reviewers understand why an activity may warrant further investigation.

- **Risk Assessment**
  - Produces a risk score based on the application's configured analysis logic.
  - Organizes findings to help reviewers prioritize suspicious activity.

- **Investigation Reports**
  - Downloadable Markdown reports.
  - Downloadable CSV reports.

- **Investigation History**
  - Uses SQLite to maintain local application investigation history in the deployment environment.

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
   Choose one of the built-in datasets representing normal activity, suspicious indicators, or ambiguous activity.

2. **Choose an operating mode**  
   Run the investigation through the Groq-powered LLM agent or use the rule-based demo mode.

3. **Analyze the available data**  
   Python analysis tools process the transaction data and available login-event information using the application's configured rules.

4. **Coordinate the investigation**  
   In LLM Agent Mode, the agent can select from permitted tools and use their outputs to formulate an investigation summary.

5. **Review findings and evidence**  
   Examine the findings, risk score, and supporting evidence identifiers. The results should be reviewed in context rather than treated as definitive proof of fraud.

6. **Export and retain results**  
   Download the available investigation reports and review saved investigation history.

### High-Level Architecture

```text
                 ┌──────────────────────────┐
                 │      Streamlit UI        │
                 │ Scenario & Mode Selection│
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │   Investigation Flow     │
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

*Conceptual architecture illustrating the main application flow.*

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and analysis logic |
| Streamlit | Interactive web interface |
| Groq API | Language-model access for LLM Agent Mode |
| LangGraph | Agent workflow orchestration |
| Pandas | Tabular data processing |
| SQLite | Investigation history storage |
| CSV | Built-in datasets and report export |
| GitHub | Source-code hosting and version control |
| Streamlit Community Cloud | Application deployment |

---

## 📂 Project Structure

```text
VerifAI/
├── app.py                  # Streamlit application
├── agent.py                # Agent workflow
├── groq_service.py         # Groq API integration
├── tools.py                # Investigation and analysis tools
├── database.py             # SQLite history management
├── report_generator.py     # Report generation
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test configuration
├── .env.example            # Environment variable template
├── assets/                 # Application logo and visual assets
├── data/                   # Built-in investigation datasets
├── docs/                   # Project documentation
└── tests/                  # Automated tests
```

---

## ⚙️ Run Locally

### Prerequisites

- Python installed on your system.
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

For local development, create a `.env` file using `.env.example` as a reference, and add your key according to the configuration expected by the application.

Keep `.env` out of version control. Never commit API keys to GitHub.

### 5. Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser.

**Rule-Based Demo Mode** can be used without making LLM calls. LLM Agent Mode requires valid Groq configuration.

---

## ☁️ Deployment

VerifAI is deployed using **Streamlit Community Cloud** and its source code is hosted on GitHub.

**Live application:** [https://verifai-fraud-investigation.streamlit.app/](https://verifai-fraud-investigation.streamlit.app/)

For your own deployment:

1. Push the project files to a GitHub repository.
2. Connect the repository to Streamlit Community Cloud.
3. Set the main file path to `app.py`.
4. Add required secrets in the app's Streamlit settings.

For Groq integration, the secret is configured in TOML format:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Replace the example value with your own key. Do not put real secrets in this README or in source code.

---

## 🔐 Security and Responsible Use

- **Protect API credentials:** Store secrets in Streamlit Secrets or appropriate environment variables.
- **Use synthetic demonstration data:** The built-in scenarios are intended for exploring application functionality, not for validating the system against real-world fraud cases.
- **Review evidence carefully:** Suspicious indicators are signals for further investigation, not proof of wrongdoing.
- **Keep human oversight:** Financial decisions should involve appropriate human review and established organizational controls.
- **Understand deployment persistence:** Local SQLite files on cloud hosting may not provide permanent storage across restarts or redeployments unless persistent storage is configured.

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
- More comprehensive automated testing and evaluation against labeled datasets.
- Persistent cloud storage and access controls for multi-user deployments.

These are potential enhancements, not features currently claimed as implemented.

---

## 👩‍💻 Author

**Anindita Bandopadhyaya**

Developer of VerifAI — Intelligent Financial Fraud Investigation.

- **GitHub:** [@anindita1807](https://github.com/anindita1807)
- **Live Project:** [Launch VerifAI](https://verifai-fraud-investigation.streamlit.app/)

---

## 📄 License

No license has been specified for this repository yet. If you plan to distribute or reuse the project publicly, consider adding an appropriate open-source license after reviewing the dependencies and ownership requirements.

---

<p align="center">
  <b>VerifAI — Turning transaction signals into structured, evidence-driven investigations.</b>
</p>
