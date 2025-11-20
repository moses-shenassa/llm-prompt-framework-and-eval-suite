# Data Folder

This folder contains small, illustrative datasets used by the evaluation harness.

## Structure

```text
data/
  datasets/
    classification_examples.jsonl
    summarization_examples.jsonl
    extraction_examples.jsonl
  gold_labels/
    classification_gold.jsonl
    summarization_gold.jsonl
    extraction_gold.jsonl
```

- `datasets/` – input examples for each task.
- `gold_labels/` – corresponding gold-standard outputs for scoring.

See `docs/dataset_description.md` for full details on formats and fields.

> In a real deployment, you would likely:
> - Replace these toy datasets with domain-specific examples.
> - Increase the number of examples per task (e.g., 50–500+).
> - Separate public sample data from private production data.
