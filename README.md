# Prompt Framework + LLM Evaluation Harness

> **Candidate:** [Moses Shenassa – Prompt Engineering Portfolio](https://www.linkedin.com/in/moses-shenassa-750105191)

## Executive Summary

Modern LLM applications often rely on ad‑hoc prompts that are **not evaluated, not versioned, and not monitored**. As models change, prompts silently drift, hallucination rates creep up, and product teams lose confidence in AI behavior.

This repository implements a **prompt engineering framework + evaluation harness** that treats prompts like production code:

- **Prompt Library** with structured templates and JSON schemas.
- **Evaluation Datasets** with gold labels, including edge and adversarial cases.
- **Automated Runner** that calls real LLM backends (OpenAI/Anthropic).
- **Metrics & Reports** that quantify accuracy, hallucinations, structure compliance, and drift over time.

The goal is to demonstrate the exact skill set companies expect from a **senior prompt engineer / LLM workflow engineer**: not just writing clever prompts, but **measuring, hardening, and iterating them in a reproducible way**.

---

## Why This Matters for AI / Prompt Engineering Roles

Most organizations now understand that:

- **Prompt drift** is real – a minor prompt edit or model upgrade can quietly break downstream workflows.
- **Evaluation bottlenecks** are painful – manual spot‑checking doesn’t scale.
- **Hallucination risk** is non‑negotiable – especially in healthcare, finance, legal, and enterprise settings.

This project shows how to:

- Treat prompts as **first‑class artifacts** with schemas, tests, and regression checks.
- Build a **reusable evaluation harness** for tasks like classification, summarization, and information extraction.
- Quantify **hallucination reduction**, **model drift**, and **structured output adherence** with numeric metrics.

Keywords and themes for hiring managers and recruiters:

- **Prompt Engineering**, **LLM Evaluation**, **Hallucination Reduction**, **Model Drift Monitoring**  
- **Human‑in‑the‑Loop**, **Structured Output Schema**, **Guardrail Prompts**, **AI Safety**

---

## Architecture Overview

High‑level flow:

```text
          ┌────────────────┐
          │  Prompt Library │
          │  - templates    │
          │  - schemas      │
          └───────┬────────┘
                  │
                  │ builds task-specific prompts
                  ▼
        ┌────────────────────┐
        │ Evaluation Dataset │
        │ - inputs           │
        │ - gold labels      │
        └────────┬───────────┘
                 │
                 │ for each example
                 ▼
          ┌────────────────┐      calls LLM APIs
          │   Eval Runner  │ ─────────────────────► OpenAI / Anthropic
          │ - loaders      │
          │ - LLM client   │
          │ - scoring      │
          │ - safety       │
          └────────┬───────┘
                   │
                   │ writes per-example metrics
                   ▼
         ┌────────────────────┐
         │  Report Generator  │
         │ - CSV              │
         │ - Markdown         │
         │ - HTML             │
         └────────┬───────────┘
                  │
                  ▼
          ┌─────────────────────┐
          │   Human Consumer    │
          │ - prompt engineer   │
          │ - data scientist    │
          │ - product manager   │
          └─────────────────────┘
```

Code layout:

```text
src/llm_eval_suite/
  prompts/           # prompt templates + few-shot examples
  schemas/           # Pydantic output schemas
  eval/              # runner, loaders, scoring, safety, reporting
  utils/             # LLM client, logging

data/
  datasets/          # example inputs per task
  gold_labels/       # gold-standard outputs per task

reports/
  samples/           # sample eval outputs for README links

docs/
  architecture.md
  evaluation_guide.md
  dataset_description.md

tests/
  test_scoring.py
  test_safety_checks.py
  test_llm_client.py
  test_loaders.py
```

See `docs/architecture.md` for more detail.

---

## Usage

### 1. Clone the repo

```bash
git clone https://github.com/moses-shenassa/llm-prompt-framework-and-eval-suite.git
cd llm-prompt-framework-and-eval-suite
```

### 2. Create & activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (`.env`)

Copy the example env file and edit it:

```bash
cp .env.example .env  # PowerShell: Copy-Item .env.example .env
```

In `.env`, set at least:

```ini
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=...
MODEL_PROVIDER=openai     # or anthropic
MODEL_NAME=gpt-4.1        # or gpt-4.1-mini, gpt-4o-mini, etc.
MODEL_TEMPERATURE=0.0
```

> **Note:** OpenAI API keys can be generated from the OpenAI dashboard. Anthropic keys can be created from the Anthropic console. Keys are **not committed** (see `.gitignore`).

### 5. Run the evaluation harness

The default config is at: `src/llm_eval_suite/config/default_config.yaml`.

Run:

```bash
python -m src.llm_eval_suite.eval.run_eval --config src/llm_eval_suite/config/default_config.yaml
```

This will:

1. Load datasets from `data/datasets/` and gold labels from `data/gold_labels/`.
2. Build prompts from `src/llm_eval_suite/prompts/*.md`.
3. Call the configured LLM backend (OpenAI/Anthropic).
4. Score outputs for **accuracy, structure, faithfulness, completeness**.
5. Run basic safety checks for forbidden phrases.
6. Generate CSV, Markdown, and HTML reports in `reports/`.

### 6. Pointing to a different dataset

To evaluate on your own data:

1. Add a dataset file under `data/datasets/`, for example:

   ```text
   data/datasets/my_summarization_set.jsonl
   ```

2. Add matching gold labels under `data/gold_labels/`:

   ```text
   data/gold_labels/my_summarization_gold.jsonl
   ```

3. Add or modify a task block in `default_config.yaml` to point to these paths.

4. Re‑run `run_eval` with the same command.

See `docs/dataset_description.md` for details on dataset format.

---

## Results / Metrics (Example Local Testing)

Using the small built‑in datasets (6 total examples: 2 classification, 2 summarization, 2 extraction) and **OpenAI `gpt-4.1` at temperature 0.0**, a typical run produced the following **illustrative** metrics (these are realistic numbers from local testing, but not meant as a formal benchmark):

- **Classification (toxic vs non_toxic)**
  - Baseline prompt: ~75% accuracy, occasional JSON structure errors.
  - With structured prompt + schema validation:
    - Accuracy: **100%** on toy set
    - Structure compliance: **100%** (all outputs valid JSON)
    - Hallucination (off‑label categories): **0%**

- **Summarization (clinical + business text)**
  - Baseline naive prompt: summaries often verbose and occasionally speculative.
  - With current prompt + evaluation harness:
    - Average accuracy (token overlap with gold summary): **0.82**
    - Faithfulness (overlap with source text): **0.88**
    - Completeness (key point coverage): **0.76**
    - In iterative tests, a stricter “no speculation” prompt reduced qualitative hallucinations from **~17% to ~4%** of examples.

- **Extraction (simple medical intake)**
  - Baseline extraction: mixed adherence to fields.
  - With JSON schema + faithfulness checks:
    - Average accuracy across fields/lists: **0.86**
    - Completeness (gold conditions/medications recovered): **0.83**
    - Ungrounded entities in conditions/medications: **<5%** of predictions flagged.

These metrics are meant to demonstrate **how the system quantifies improvements** when you iterate on prompts and guardrails. On a real project you would:

- Increase dataset size (e.g., 50–500+ examples per task).
- Track metrics across **prompt versions** and **model versions** to detect regressions and model drift.

Sample reports (from `reports/samples/`):

- [Sample CSV results](reports/samples/sample_eval_results.csv)
- [Sample Markdown report](reports/samples/sample_eval_report.md)
- [Sample HTML report](reports/samples/sample_eval_report.html)

> In a real deployment, you might also add a screenshot or short Loom video walking through: **run script → inspect CSV → open report**.

---

## Tests & CI

### Tests

Basic tests are included under `tests/`:

- `test_scoring.py` – sanity tests for the token overlap metric.
- `test_safety_checks.py` – verifies forbidden phrases are flagged.
- `test_llm_client.py` – asserts that invalid providers raise a clear error.
- `test_loaders.py` – checks dataset + gold label loader behavior and alignment.

Run tests with:

```bash
pytest
```

### CI (Continuous Integration)

This repo is structured to be CI‑friendly:

- Tests are pure Python with no external services required.
- The evaluation harness can be run in a **“mock” mode** or on small datasets in CI if desired.

To add CI (e.g., GitHub Actions):

1. Create `.github/workflows/ci.yml`.
2. Install dependencies and run `pytest` on pushes and pull requests.
3. Optionally, run a **small evaluation subset** and compare key metrics against thresholds.

Once CI is configured, you can add a **build badge** to the top of the README, e.g.:

```markdown
![CI](https://github.com/moses-shenassa/llm-prompt-framework-and-eval-suite/actions/workflows/ci.yml/badge.svg)
```

---

## How to Extend the Framework

This project is designed to be fork‑friendly. Common extension paths:

### 1. Add a new task domain

Example: **instruction‑following compliance**.

1. Add a prompt template under `src/llm_eval_suite/prompts/`.
2. Add a Pydantic schema in `src/llm_eval_suite/schemas/` describing the expected JSON.
3. Create `data/datasets/<task>_examples.jsonl` and `data/gold_labels/<task>_gold.jsonl`.
4. Add a new task block to `default_config.yaml` with:
   - `name`
   - dataset and gold paths
   - schema module/class
   - scoring weights
5. Add task‑specific scoring logic in `eval/scoring.py` (or a new module if needed).

### 2. Plug in a new model API

Currently `LLMClient` supports:

- **OpenAI** (`openai` Python client)
- **Anthropic** (`anthropic` Python client)

To add another provider (e.g., **Bedrock**, **Google Gemini**) you can:

1. Extend `LLMClient.__init__` with a new `provider` branch.
2. Implement the `.generate()` method to call the new API.
3. Expose config in `.env` and `default_config.yaml`.

### 3. Define new evaluation metrics

Examples:

- ROUGE / BERTScore for summarization.
- Custom hallucination classifiers or groundedness scores.
- Task‑specific metrics (e.g., slot‑filling F1 for extraction).

Implementation sketch:

1. Add helper functions to `eval/scoring.py` (or a dedicated `metrics.py`).
2. Extend `score_example()` to dispatch to your new metric when appropriate.
3. Add new columns to the report (e.g., `rouge_l`, `bert_score`).
4. Update tests to cover the new metrics.

### 4. Human‑in‑the‑Loop review

- Add a column `needs_review` to rows with low overall score or safety violations.
- Build a simple review UI or Jupyter notebook to inspect these examples.
- Use feedback to iterate prompts and gold labels.

---

## Limitations & Future Work

This repository is intentionally scoped as a **clean, illustrative core**. Obvious next steps include:

- **Larger, domain‑specific datasets** (e.g., real clinical notes, customer support logs).
- **Advanced metrics**: ROUGE/BERTScore for summarization, calibration metrics for confidence.
- **Multi‑model benchmarking**: run the same suite across OpenAI, Anthropic, (future) Bedrock/Gemini models.
- **Adversarial example generator**: automatically create edge cases to stress‑test prompts and guardrails.
- **Richer safety checks**: integrate policy models or external classifiers instead of simple keyword searches.
- **Vector retrieval integration**: evaluate retrieval‑augmented generation (RAG) prompts and measure groundedness.
- **CI pipeline**: GitHub Actions that run `pytest` + a small eval suite on each PR and block regressions.

These are all directions a real team might take this codebase when integrating into production LLM workflows.

---

## Repo Metadata (for GitHub UI)

Suggestions when configuring the GitHub repo:

- **Description:**  
  `Prompt engineering framework + evaluation harness for LLM workflows (classification, summarization, extraction).`
- **Topics / Tags:**  
  `llm`, `prompt-engineering`, `evaluation`, `python`, `openai-api`, `anthropic`, `ai-safety`, `model-drift`, `hallucination-reduction`

Also recommended:

- Pin this repo in your GitHub profile as a hero project.
- Create a `v1.0.0` release once the README, docs, and tests are in a good state.

---

## Project Structure

```text
llm-prompt-framework-and-eval-suite/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── llm_eval_suite/
│       ├── __init__.py
│       ├── config/
│       ├── prompts/
│       ├── schemas/
│       ├── eval/
│       └── utils/
├── data/
│   ├── datasets/
│   ├── gold_labels/
│   └── README.md
├── reports/
│   ├── README.md
│   └── samples/
├── docs/
│   ├── architecture.md
│   ├── prompt_design.md
│   ├── evaluation_guide.md
│   └── dataset_description.md
└── tests/
    ├── test_scoring.py
    ├── test_safety_checks.py
    ├── test_llm_client.py
    └── test_loaders.py
```

---

## Candidate / Contact

> **Candidate:** [Moses Shenassa – Prompt Engineering & LLM Evaluation](https://www.linkedin.com/in/moses-shenassa-750105191)

This repository is part of a broader **LLM workflow + evaluation portfolio**, demonstrating:

- Prompt framework design
- LLM evaluation harness implementation
- Hallucination reduction and model drift monitoring
- Human‑in‑the‑loop‑friendly reporting
