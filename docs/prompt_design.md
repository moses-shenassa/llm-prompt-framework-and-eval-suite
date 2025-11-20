# Prompt Design

Each task has:

- A **base prompt** in `src/llm_eval_suite/prompts/` that defines:
  - The task (classification, summarization, extraction).
  - The required JSON output schema.
  - Guardrail instructions (no extra commentary, no speculation, etc.).

- **Few-shot examples** in `src/llm_eval_suite/prompts/few_shot_examples/`:
  - Simple (input, output) pairs that demonstrate the desired behavior.
  - These can be plugged into prompts or used in separate evaluation tasks.

- A **Pydantic schema** in `src/llm_eval_suite/schemas/`:
  - Used to validate that the model output conforms to the expected structure.
