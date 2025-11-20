# Architecture Overview

This project is organized into three main layers:

1. **Prompt Library**
2. **Evaluation Harness**
3. **Reporting & Metrics**

## Prompt Library

Located in `src/llm_eval_suite/prompts/` and `src/llm_eval_suite/schemas/`.

- **Base prompt templates** (Markdown):
  - `base_classification_prompt.md`
  - `base_summarization_prompt.md`
  - `base_extraction_prompt.md`
- **Few-shot examples** (JSON):
  - `few_shot_examples/*.json`
- **Structured output schemas** (Pydantic models):
  - `schemas/classification_schema.py`
  - `schemas/summarization_schema.py`
  - `schemas/extraction_schema.py`

Prompts explicitly describe the **output JSON schema**, so we can validate and score LLM outputs reliably.

## Evaluation Harness

Located in `src/llm_eval_suite/eval/` and `src/llm_eval_suite/utils/`.

Key components:

- **Config Loader**
  - `config/default_config.yaml`
  - Defines tasks, datasets, schemas, scoring weights, output paths.

- **Dataset Loader**
  - `eval/loaders.py`
  - Loads JSONL datasets and gold labels.
  - Aligns examples and gold rows by ID.

- **LLM Client**
  - `utils/llm_client.py`
  - Wraps OpenAI / Anthropic clients behind a single `.generate()` method.
  - Reads provider + model from config.

- **Scoring & Metrics**
  - `eval/scoring.py`
  - Task-specific metrics for:
    - Classification
    - Summarization
    - Extraction
  - Computes accuracy, structure compliance, faithfulness, completeness.
  - `score_example()` is the high-level entry point used by the runner.

- **Safety Checks**
  - `eval/safety_checks.py`
  - Keyword-based safety heuristics (forbidden phrases).
  - Designed to be swapped out for more advanced safety modules.

- **Runner**
  - `eval/run_eval.py`
  - Orchestrates the full evaluation:
    - Reads config.
    - Iterates over tasks and examples.
    - Builds prompts from templates.
    - Calls the LLM client.
    - Scores outputs and safety.
    - Collects per-example rows and passes them to reporting.

## Reporting & Metrics

Located in `eval/reporting.py` and `reports/`.

- Writes a **CSV** with per-example metrics.
- Writes a **Markdown** report with aggregate stats per task.
- Wraps the Markdown as a simple **HTML** report for easy viewing.
- Sample outputs live in `reports/samples/` and are linked from `README.md`.

The reporting layer is intentionally simple so it can be:

- Dropped into dashboards, or
- Extended with better HTML/JS front-ends, or
- Read directly by data scientists in notebooks.
