# Prompt Design

Each task in this project is backed by an explicit prompt template and schema.

## Components

1. **Base Prompt Template** (Markdown)
   - Describes the task (classification, summarization, extraction).
   - Specifies the required JSON output structure.
   - Includes guardrail instructions (no speculation, no extra commentary, etc.).

2. **Few-shot Examples** (JSON)
   - Simple `(input, output)` pairs stored in `prompts/few_shot_examples/*.json`.
   - Can be embedded into prompts or used to sanity check behavior.

3. **Pydantic Schema**
   - Defines the structured output expected from the LLM.
   - Parsed and validated at evaluation time.
   - Allows us to treat malformed outputs as explicit failures.

## Goals

- Make prompt contracts **explicit and enforceable**.
- Encourage **structured outputs** over free-form text.
- Provide a stable foundation for **prompt iteration and regression testing**.
