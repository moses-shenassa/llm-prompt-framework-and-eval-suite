# Evaluation Guide

The evaluation harness computes several metrics per example:

- **Accuracy** – task-specific measure of correctness.
- **Structure** – whether the output matches the expected schema.
- **Faithfulness** – how well the output aligns with the source text.
- **Completeness** – how much of the required information is captured.

Each task can weight these metrics differently via `config/default_config.yaml`.

## Running Evaluations

Use:

```bash
python -m src.llm_eval_suite.eval.run_eval --config src/llm_eval_suite/config/default_config.yaml
```

This will:

1. Load all enabled tasks.
2. Call the configured LLM backend.
3. Score each example.
4. Write CSV + Markdown + HTML reports into `reports/`.

## Drift and Regression

Because prompts, schemas, and datasets are versioned together, you can:

- Re-run evaluations after changing prompts or models.
- Compare average scores to detect regressions.
- Run on multiple models to compare behavior using the same tests.

In a CI setting, you can:

- Run a small subset on each PR.
- Block merges if key metrics (e.g., overall score, structure compliance) fall below thresholds.
