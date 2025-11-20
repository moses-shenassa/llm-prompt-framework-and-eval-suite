"""
Lightweight Streamlit front end for the LLM Prompt Framework + Evaluation Harness.

Goal:
- Allow non-technical users to paste or upload text.
- Run it through the configured LLM (OpenAI/Anthropic).
- Validate structured outputs (JSON schemas) and run safety checks.
- Show actionable, human-readable metrics.

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from llm_eval_suite.utils.llm_client import LLMClient
from llm_eval_suite.eval.safety_checks import run_safety_checks

# These imports assume you have Pydantic schemas defined as in the docs.
# If your class names / paths differ, update these imports accordingly.
from llm_eval_suite.schemas.classification_schema import ClassificationOutput
from llm_eval_suite.schemas.summarization_schema import SummarizationOutput
from llm_eval_suite.schemas.extraction_schema import ExtractionOutput

# Optional faithfulness proxy for summarization
try:
    from llm_eval_suite.eval.scoring import _token_overlap
except ImportError:
    _token_overlap = None  # Fallback if not available


# ----- Helpers ----- #

@dataclass
class EvalResult:
    task: str
    input_text: str
    raw_output: str
    parsed_output: Optional[Dict[str, Any]]
    structure_valid: bool
    safety_violated: bool
    safety_reasons: List[str]
    faithfulness: Optional[float] = None  # summarization only (if available)


def build_prompt(task: str, input_text: str) -> str:
    """
    Very simple task-specific prompt builder.

    For production use, you might want to:
    - Load templates from src/llm_eval_suite/prompts/
    - Embed few-shot examples, etc.

    For this lightweight front end, we keep it simple but explicit.
    """
    if task == "classification":
        return (
            "You are a careful content classifier.\n\n"
            "Task: Classify the following text as 'toxic' or 'non_toxic'.\n"
            "Output a JSON object with this exact schema:\n"
            '{ "label": "toxic" | "non_toxic", "reason": "short explanation" }\n\n'
            "Text to classify:\n"
            f"{input_text}\n"
        )
    elif task == "summarization":
        return (
            "You are a precise summarization assistant.\n\n"
            "Task: Write a short, faithful summary of the text below.\n"
            "Do not speculate beyond the text. Do not introduce new facts.\n"
            "Output a JSON object with this exact schema:\n"
            '{ "summary": "short paragraph", "key_points": ["bullet 1", "bullet 2", ...] }\n\n'
            "Text to summarize:\n"
            f"{input_text}\n"
        )
    elif task == "extraction":
        return (
            "You are an information extraction assistant.\n\n"
            "Task: Extract structured information from the following text.\n"
            "If a field is unknown, use null or an empty list.\n"
            "Output a JSON object with this exact schema:\n"
            '{\n'
            '  "name": "string or null",\n'
            '  "age": 32 or null,\n'
            '  "location": "string or null",\n'
            '  "conditions": ["list", "of", "conditions"],\n'
            '  "medications": ["list", "of", "medications"]\n'
            '}\n\n'
            "Text to extract from:\n"
            f"{input_text}\n"
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_output(task: str, raw_output: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Try to parse the model output as JSON using the appropriate Pydantic schema.
    Returns (parsed_object_or_None, structure_valid_flag).
    """
    raw_output = raw_output.strip()

    # Try to locate a JSON object in the output, if the model added extra text.
    json_str = raw_output
    if not raw_output.startswith("{"):
        # naive heuristic: find first '{' and last '}' and slice
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw_output[start : end + 1]

    try:
        if task == "classification":
            obj = ClassificationOutput.model_validate_json(json_str)
        elif task == "summarization":
            obj = SummarizationOutput.model_validate_json(json_str)
        elif task == "extraction":
            obj = ExtractionOutput.model_validate_json(json_str)
        else:
            raise ValueError(f"Unsupported task: {task}")
        return obj.model_dump(), True
    except Exception:
        return None, False


def faithfulness_proxy(source_text: str, summary_text: str) -> Optional[float]:
    """
    Use token overlap between summary and source as a rough faithfulness proxy.

    If _token_overlap is not available, returns None.
    """
    if _token_overlap is None:
        return None
    return _token_overlap(source_text, summary_text)


def run_quick_eval(
    task: str,
    inputs: List[str],
    model_provider: str,
    model_name: str,
    temperature: float,
) -> List[EvalResult]:
    """
    Core function used by the UI:
    - Builds prompts
    - Calls LLM
    - Parses outputs
    - Runs safety checks
    - Computes simple metrics
    """

    client = LLMClient(
        provider=model_provider,
        model_name=model_name,
        temperature=temperature,
    )

    results: List[EvalResult] = []

    for text in inputs:
        prompt = build_prompt(task, text)

        # Build system + user prompts for LLMClient.generate(system, user)
        system_prompt = (
            "You are a structured-output assistant. "
            "Follow the JSON schema exactly. "
            "Do not add commentary or explanation outside the JSON object."
        )

        # Call your actual LLM client with the correct signature
        raw_output = client.generate(system_prompt, prompt)

        parsed, structure_valid = parse_output(task, raw_output)

        # Run safety checks on BOTH the user input and model output.
        # This way we can detect risky user prompts (self-harm, violence, etc.)
        # even if the model returns a neutral/structured JSON object.
        combined_text = f"USER INPUT:\n{text}\n\nMODEL OUTPUT:\n{raw_output}"
        safety = run_safety_checks(combined_text)

        safety_violated = safety.has_violation
        safety_reasons = safety.reasons
        faith = None
        if task == "summarization" and parsed is not None:
            summary_text = parsed.get("summary", "")
            faith = faithfulness_proxy(text, summary_text)

        results.append(
            EvalResult(
                task=task,
                input_text=text,
                raw_output=raw_output,
                parsed_output=parsed,
                structure_valid=structure_valid,
                safety_violated=safety_violated,
                safety_reasons=safety_reasons,
                faithfulness=faith,
            )
        )

    return results


# ----- Streamlit UI ----- #

def main() -> None:
    st.set_page_config(
        page_title="Prompt Framework – Quick Eval UI",
        layout="wide",
    )

    st.title("Prompt Framework + LLM Evaluation Harness – Quick Eval UI")
    st.markdown(
        """
This interface is designed for **non-technical users** to quickly try out prompts and inputs
against the underlying LLM + guardrail stack, without touching code or datasets.

- Paste one or more texts to analyze (one per line), **or** upload a file.
- Choose a task (classification, summarization, extraction).
- Click **Run evaluation** to see structured outputs, safety flags, and simple metrics.

> Note: For full benchmark-style evaluations (with gold labels and detailed metrics),
> use the command-line harness described in the main README.
"""
    )

    st.sidebar.header("Model & Settings")

    model_provider = st.sidebar.selectbox(
        "Provider",
        options=["openai", "anthropic"],
        index=0,
        help="Configured via your .env and underlying LLMClient.",
    )
    model_name = st.sidebar.text_input(
        "Model name",
        value="gpt-4.1",
        help="Example: gpt-4.1, gpt-4.1-mini, gpt-4o-mini. Must be supported by your provider.",
    )
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values = more creative, lower values = more deterministic.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Candidate:** [Moses Shenassa](https://www.linkedin.com/in/moses-shenassa-750105191)")

    st.subheader("1. Choose Task")

    task = st.selectbox(
        "Task type",
        options=[
            ("classification", "Classification (toxic vs non_toxic)"),
            ("summarization", "Summarization"),
            ("extraction", "Information Extraction"),
        ],
        format_func=lambda x: x[1],
    )[0]

    st.subheader("2. Provide Inputs")

    input_mode = st.radio(
        "Input mode",
        options=["Text area (one per line)", "Upload file (CSV or JSONL)"],
    )

    user_inputs: List[str] = []

    if input_mode == "Text area (one per line)":
        raw_text = st.text_area(
            "Enter one example per line",
            height=200,
            placeholder="Example 1...\nExample 2...\nExample 3...",
        )
        if raw_text.strip():
            user_inputs = [line.strip() for line in raw_text.splitlines() if line.strip()]
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or JSONL file",
            type=["csv", "jsonl"],
            help="CSV: first column used as input. JSONL: expects objects with an 'input' field.",
        )
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(uploaded_file)
                if len(df.columns) == 0:
                    st.error("CSV has no columns.")
                else:
                    # For simplicity, use the first column as input text.
                    col = df.columns[0]
                    user_inputs = df[col].dropna().astype(str).tolist()
                    st.info(f"Loaded {len(user_inputs)} rows from column '{col}'.")
            elif uploaded_file.name.endswith(".jsonl"):
                lines = uploaded_file.read().decode("utf-8").splitlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("input") or obj.get("text")
                        if text:
                            user_inputs.append(str(text))
                    except json.JSONDecodeError:
                        continue
                st.info(f"Loaded {len(user_inputs)} rows from JSONL file.")
            else:
                st.error("Unsupported file type. Please upload CSV or JSONL.")

    st.subheader("3. Run")

    if st.button("Run evaluation", type="primary"):
        if not user_inputs:
            st.warning("Please provide at least one input via the text area or upload a valid file.")
            return

        with st.spinner("Calling LLM and evaluating outputs..."):
            try:
                results = run_quick_eval(
                    task=task,
                    inputs=user_inputs,
                    model_provider=model_provider,
                    model_name=model_name,
                    temperature=temperature,
                )
            except Exception as e:
                st.error(f"Error while running evaluation: {e}")
                return

        # Aggregate and show metrics
        total = len(results)
        structure_valid_count = sum(1 for r in results if r.structure_valid)
        safety_violation_count = sum(1 for r in results if r.safety_violated)
        faithfulness_scores = [
            r.faithfulness for r in results if r.faithfulness is not None
        ]

        st.success("Evaluation completed.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total examples", total)
        col2.metric(
            "Structured output rate",
            f"{structure_valid_count}/{total} ({structure_valid_count / total:.0%})",
        )
        col3.metric(
            "Safety violations detected",
            f"{safety_violation_count}/{total} ({safety_violation_count / total:.0%})",
        )

        if task == "summarization" and faithfulness_scores:
            avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
            st.metric(
                "Average faithfulness (token overlap proxy)",
                f"{avg_faith:.2f}",
                help="Higher is better; 1.0 means all summary tokens appear in source text.",
            )

        st.markdown("---")
        st.subheader("Per-example Results")

        for idx, r in enumerate(results, start=1):
            with st.expander(f"Example {idx}"):
                st.markdown("**Input text:**")
                st.code(r.input_text, language="text")

                st.markdown("**Raw model output:**")
                st.code(r.raw_output, language="json")

                st.markdown("**Parsed structured output (if valid):**")
                if r.parsed_output is not None:
                    st.json(r.parsed_output)
                else:
                    st.write("_Parsing failed – output did not match the expected JSON schema._")

                cols = st.columns(3)
                cols[0].write(f"**Structure valid:** {'✅' if r.structure_valid else '❌'}")
                cols[1].write(f"**Safety violation:** {'⚠️' if r.safety_violated else '✅ None'}")

                if r.faithfulness is not None:
                    cols[2].write(f"**Faithfulness (proxy):** {r.faithfulness:.2f}")
                else:
                    cols[2].write("**Faithfulness:** n/a")

                if r.safety_reasons:
                    st.markdown("**Safety notes:**")
                    for reason in r.safety_reasons:
                        st.write(f"- {reason}")


if __name__ == "__main__":
    main()
