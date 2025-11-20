# Prompt Framework + LLM Evaluation Harness

## Executive Summary

This repository provides a **prompt engineering framework plus evaluation harness** for large language models (LLMs). It lets you define reusable prompt templates, run structured evaluations over labeled datasets, measure reliability and hallucinations, and generate human-readable reports — with a simple Streamlit UI for non-technical users.

Most prompt engineering in the wild is ad-hoc and brittle. This project demonstrates how to turn prompts into **versioned, testable, measurable artifacts**, exactly the kind of capability hiring teams now expect from modern prompt engineers.

---

## Architecture Overview

At a high level, the system looks like this:

```text
                 ┌─────────────────────────────┐
                 │        Prompt Library       │
                 │  (templates + guardrails,   │
                 │   Pydantic-style schemas)   │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │      Evaluation Datasets    │
                 │  (inputs + gold labels,     │
                 │   edge & adversarial cases) │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │       Eval Runner CLI       │
                 │  (reads config, calls LLM   │
                 │   via provider abstraction, │
                 │   computes metrics)         │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │       Metric Modules        │
                 │  (accuracy, structure,      │
                 │   hallucination, drift)     │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │      Report Generator       │
                 │  (CSV + Markdown + HTML)    │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │        Streamlit UI         │
                 │  (non-technical users can   │
                 │   upload data, run evals,   │
                 │   inspect safety flags)     │
                 └─────────────────────────────┘
```

**Key concepts:**

- **Prompt templates** – reusable instructions for classification, extraction, and summarization, with clearly documented input variables and output expectations.
- **Provider abstraction** – a thin client layer that hides whether calls go to OpenAI, Anthropic, etc.
- **Config structure** – YAML configs tie together models, datasets, prompts, metrics, and output paths.
- **Metric modules** – Python functions that compute accuracy, structure compliance, and hallucination proxies.
- **Eval runner** – command-line entry point that loads config, runs the LLM over the dataset, and writes reports.
- **Report generation** – CSV for raw scores, Markdown + HTML for human-readable reports.
- **Streamlit UI** – simple web app so product/safety/ops folks can try prompts and see safety flags without touching Python.

---

## Quickstart

This quickstart assumes **Python 3.10+** and `git` are installed.

```bash
# 1. Clone the repository
git clone https://github.com/moses-shenassa/llm-prompt-framework-and-eval-suite.git
cd llm-prompt-framework-and-eval-suite

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# then edit .env and add your API key, for example:
# OPENAI_API_KEY=sk-...

# 5. Run a full classification evaluation (support ticket example)
python scripts/run_eval.py --config configs/support_ticket_classification.yaml
```

This will:

1. Load the support ticket classification config.
2. Call the chosen LLM via the provider abstraction.
3. Evaluate model outputs against gold labels.
4. Write a CSV + Markdown + HTML report under `reports/`.

---

## Example End-to-End Task: Support Ticket Classification

A concrete, fully wired example is included to make the framework “real” for reviewers.

### Task

> Take short customer support tickets and classify them into categories like *billing*, *technical_issue*, *account_access*, etc.

### Data

`data/support_tickets.csv` contains columns like:

- `id` – unique identifier
- `text` – raw support ticket text
- `label` – gold-standard category

Example rows:

```csv
id,text,label
1,"I was charged twice this month for the same subscription.",billing
2,"The app crashes every time I try to upload a file.",technical_issue
3,"I can't log into my account even after resetting my password.",account_access
```

### Config

`configs/support_ticket_classification.yaml` binds everything together:

- Model provider + model name  
- Path to the dataset  
- Prompt template to use  
- Output schema (classification JSON)  
- Metrics to compute (accuracy, structure validity)  
- Report output directory  

You can run it with:

```bash
python scripts/run_eval.py --config configs/support_ticket_classification.yaml
```

### Example Output (Realistic Sample)

A Markdown snippet from a typical run might look like this:

```markdown
# Evaluation Summary – Support Ticket Classification

- Task: classification
- Dataset: data/support_tickets.csv
- Model: gpt-4.1 (OpenAI)
- Examples: 50

## Aggregate Metrics

| Metric                        | Value |
|------------------------------|-------|
| Accuracy (exact label match) | 0.92  |
| Structure validity           | 0.98  |
| Avg. latency per call (s)    | 1.4   |

## Sample Row

- id: 3  
- text: "I can't log into my account even after resetting my password."  
- gold_label: `account_access`  
- model_label: `account_access`  
- structure_valid: `true`  
- safety_violations: `[]`
```

CSV and HTML versions of the same report are also generated under `reports/`.

> **Note:** The exact numbers will depend on your API key, model version, and dataset size; the above is representative of what a successful run looks like.

---

## Streamlit UI Demo (Non-Technical Friendly)

In addition to the CLI runner, this project includes a lightweight **Streamlit UI** so non-technical teammates can try prompts and see safety flags.

### Run the UI

From the repo root, with your virtualenv active:

```bash
python -m streamlit run app/streamlit_app.py
```

This opens a browser UI (usually at `http://localhost:8501`) where users can:

- Select a task: **classification**, **summarization**, or **extraction**.
- Paste multiple lines of text or upload a CSV/JSONL file.
- See, for each row:
  - Raw model output  
  - Parsed JSON (if structure is valid)  
  - Structure validity flag  
  - Safety violation flag + reasons  
  - Basic sentiment score  
  - (For summarization) a simple faithfulness proxy score  

A screenshot (see `docs/streamlit_ui.png`) can be added to the README or GitHub repo page for visual context.

---

## Safety & Guardrails

Safety is treated as a **first-class evaluation dimension**, not an afterthought.

The framework supports multi-layer safety checks on both **user inputs** and **model outputs**:

1. **Pattern-based detection**  
   - Regex rules for self-harm, violence, illegal activity, and dangerous medical behavior.

2. **Lexicon + sentiment scoring**  
   - Simple lexicon-based sentiment scoring to highlight severe emotional distress (e.g., hopeless, worthless, empty, etc.).

3. **Semantic similarity checks** (optional)  
   - OpenAI embeddings can be used to detect paraphrases of known-dangerous content even when exact keywords are not present.

4. **Structured logging**  
   - Each example includes a `safety_violations` field listing all matched rules and heuristics.

This setup makes it easy to build a **safety evaluation dataset** and to monitor the safety profile of different prompts and models over time.

> All examples and checks are designed to **model** potentially harmful prompts in order to **reject or flag** them. The system never provides instructions for self-harm, violence, or other unsafe behaviors.

---

## Extending the Framework

The project is intentionally structured to make it easy to extend.

### Add a New Task

1. **Define a dataset** in `data/` (CSV or JSONL with input + gold labels).  
2. **Create a prompt template** (e.g., `src/llm_eval_suite/prompts/my_new_task.py` or a simple Jinja-style string).  
3. **Define an output schema** (Pydantic model / dict contract) describing the expected JSON fields.  
4. **Add a config** in `configs/` describing:
   - Task type (classification/summarization/extraction/custom)
   - Dataset path
   - Prompt template reference
   - Metrics to compute  
5. **Run the eval** via:
   ```bash
   python scripts/run_eval.py --config configs/my_new_task.yaml
   ```

### Add a New Model Backend

The framework uses a provider abstraction layer, typically implemented in `src/llm_eval_suite/utils/llm_client.py`. To add a provider:

1. Implement a new client method (e.g., `call_anthropic`, `call_bedrock`, `call_gemini`).  
2. Add a `provider` value (e.g., `anthropic`) to the config and branch on it in the client.  
3. Optionally add provider-specific settings (e.g., safety filters, rate limits).  

### Add or Modify Metrics

Metrics live in a scoring module (commonly `src/llm_eval_suite/eval/scoring.py`). Adding a metric usually involves:

1. Implementing a function like `def my_new_metric(row) -> float:`.  
2. Registering it in the metric dispatch logic.  
3. Referencing it by name in the config under the task’s metrics list.

---

## Minimal Test Suite

A small but meaningful test suite is included to show professional engineering practice:

1. **Config loading test**  
   - Verifies that YAML configs are well-formed and contain required keys.

2. **Prompt template test**  
   - Ensures that the classification prompt template can be rendered with required fields (no missing variables).

3. **Metric accuracy test**  
   - Checks that the accuracy metric returns the correct value on a trivial dataset (e.g., 3/4 correct = 0.75).

4. **Provider mock test**  
   - Uses a fake provider object to simulate LLM output, so the eval pipeline can run without hitting a real API.

These tests can be run via:

```bash
pytest
```

Passing tests are a strong signal to hiring managers that this is a **deliberately engineered** system, not just a collection of scripts.

---

## Releases and Versioning

Once the repository is pushed and stable, you can create a release tag such as:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This signals that the project has reached a **coherent, reviewable milestone**, ideal for including in job applications and technical portfolios.

---

## Project Metadata

- **Author:** Moses Shenassa  
- **Focus:** Prompt engineering, LLM evaluation, safety, and drift detection  
- **Tech stack:** Python, OpenAI API (pluggable), Pandas, Streamlit  
- **Intended audience:** Hiring managers, recruiters, and technical interviewers evaluating LLM/prompt engineering skill.

If you’re reviewing this repository as part of a hiring process and would like more context, you can reach Moses via the contact information on his GitHub profile or LinkedIn.
