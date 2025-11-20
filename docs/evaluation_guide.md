# Evaluation Guide

The evaluation harness computes several metrics per example:

- **Accuracy** – task-specific measure of correctness.
- **Structure** – whether the output matches the expected schema.
- **Faithfulness** – how well the output aligns with the source text.
- **Completeness** – how much of the required information is captured.

Each task can weight these metrics differently via `default_config.yaml`.

## Drift and Regression

Because the prompts and datasets are fixed, you can:

- Re-run evaluations after changing prompts.
- Compare average scores to detect regressions.
- Run on multiple models to compare their behavior using the same tests.
