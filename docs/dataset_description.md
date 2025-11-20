# Dataset Description

This project includes small, illustrative datasets for three tasks:

1. **Classification**
2. **Summarization**
3. **Extraction**

The datasets are intentionally tiny to keep evaluation runs cheap and fast. In a real deployment, you would scale up to dozens or hundreds of examples per task.

---

## Format

All datasets are stored as **JSON Lines (JSONL)** files:

- One JSON object per line.
- No outer list.

Example (classification):

```json
{"id": "cls-1", "input": "I can't believe how stupid you are.", "allowed_labels": ["toxic", "non_toxic"]}
{"id": "cls-2", "input": "Thanks again for your help yesterday, I really appreciate it.", "allowed_labels": ["toxic", "non_toxic"]}
```

Each dataset file has a corresponding **gold labels** file, also JSONL, with the same IDs.

---

## Task: Classification

**Files**

- Dataset: `data/datasets/classification_examples.jsonl`
- Gold labels: `data/gold_labels/classification_gold.jsonl`

**Fields**

- `id`: string – unique identifier for the example.
- `input`: string – raw text to classify.
- `allowed_labels`: list of strings – labels the model is allowed to use (e.g., `"toxic"`, `"non_toxic"`).

**Gold Labels**

- `id`: string – must match dataset ID.
- `label`: string – the correct label for the input.
- `allowed_labels`: list of strings – repeated for convenience.

This task is currently configured as **toxicity detection** on toy text snippets.

---

## Task: Summarization

**Files**

- Dataset: `data/datasets/summarization_examples.jsonl`
- Gold labels: `data/gold_labels/summarization_gold.jsonl`

**Fields**

- `id`: string – unique identifier.
- `input`: string – source text to be summarized.

**Gold Labels**

- `id`: string – must match dataset ID.
- `summary`: string – short, faithful summary of the input.
- `key_points`: list of strings – bullet-style key points capturing critical details.

The examples are deliberately simple (clinical-like notes and email threads) to show how the harness works without domain complexity.

---

## Task: Extraction

**Files**

- Dataset: `data/datasets/extraction_examples.jsonl`
- Gold labels: `data/gold_labels/extraction_gold.jsonl`

**Fields**

- `id`: string – unique identifier.
- `input`: string – narrative text containing entities to extract.

**Gold Labels**

- `id`: string – must match dataset ID.
- `name`: string or `null` – person name if present.
- `age`: integer or `null` – age if present.
- `location`: string or `null` – location if present.
- `conditions`: list of strings – medical or relevant conditions.
- `medications`: list of strings – medications or treatments.

These examples mimic simple intake notes or call transcripts.

---

## Extending the Datasets

To add your own datasets:

1. Create a new JSONL file under `data/datasets/` with your examples.
2. Create a matching JSONL file under `data/gold_labels/` with gold answers.
3. Ensure each example has a unique `id` and that the gold file has a matching entry.
4. Update `src/llm_eval_suite/config/default_config.yaml` to point to your new files.

For large or sensitive datasets, you can:

- Keep only a small, synthetic subset in the repo.
- Store real data privately and mount it where the harness can read it.
