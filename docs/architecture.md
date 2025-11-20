# Architecture Overview

This project is organized around three main layers:

1. **Prompt Library**
   - Base prompt templates for each task type.
   - Few-shot examples stored as JSON files.
   - Output schemas defined with Pydantic models.

2. **Evaluation Harness**
   - Loads datasets + gold labels from `data/`.
   - Calls the configured LLM via `LLMClient`.
   - Validates outputs against schemas and scores them.
   - Runs safety checks on the model outputs.

3. **Reporting**
   - Aggregates per-example scores into summary metrics.
   - Writes CSV, Markdown, and simple HTML reports into `reports/`.
