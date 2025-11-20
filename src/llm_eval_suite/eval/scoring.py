from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError


@dataclass
class ScoreResult:
    """Container for scores per example."""

    accuracy: float
    structure: float
    faithfulness: float
    completeness: float
    overall: float
    parse_error: Optional[str] = None


def parse_with_schema(output_text: str, schema_cls: type[BaseModel]) -> BaseModel:
    """Attempt to parse model output as JSON and validate with a Pydantic schema."""
    data = json.loads(output_text)
    return schema_cls.model_validate(data)


def score_classification(
    model_output: BaseModel,
    gold: Dict[str, Any],
    weights: Dict[str, float],
) -> ScoreResult:
    """Simple scoring for classification.

    - accuracy: 1 if label matches, else 0.
    - structure: 1 (assumes schema validation already passed).
    - faithfulness: proxy: conf high when correct, lower when not.
    - completeness: always 1 (single label).
    """
    gold_label = gold["label"]
    pred_label = model_output.label

    accuracy = 1.0 if pred_label == gold_label else 0.0
    structure = 1.0
    completeness = 1.0
    faithfulness = model_output.confidence if accuracy == 1.0 else max(0.0, 1.0 - model_output.confidence)

    overall = (
        accuracy * weights.get("accuracy_weight", 0.5)
        + structure * weights.get("structure_weight", 0.2)
        + faithfulness * weights.get("faithfulness_weight", 0.2)
        + completeness * weights.get("completeness_weight", 0.1)
    )

    return ScoreResult(
        accuracy=accuracy,
        structure=structure,
        faithfulness=faithfulness,
        completeness=completeness,
        overall=overall,
    )


def _token_overlap(a: str, b: str) -> float:
    """Very simple token overlap metric for demonstration.

    Returns a value between 0 and 1.
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union)


def score_summarization(
    model_output: BaseModel,
    example: Dict[str, Any],
    gold: Dict[str, Any],
    weights: Dict[str, float],
) -> ScoreResult:
    """Simple summarization scoring.

    - accuracy: token overlap between model summary and gold summary.
    - structure: 1 if key_points non-empty, else 0.
    - faithfulness: penalize if model mentions entities not in source input.
      (Here we use a naive heuristic based on word overlap.)
    - completeness: token overlap between model.key_points joined and gold.key_points joined.
    """
    gold_summary = gold["summary"]
    model_summary = model_output.summary

    accuracy = _token_overlap(model_summary, gold_summary)

    structure = 1.0 if model_output.key_points else 0.0

    source_text = example["input"]
    # naive faithfulness: how much of model summary overlaps with source
    faithfulness = _token_overlap(model_summary, source_text)

    gold_kp_text = " ".join(gold.get("key_points", []))
    model_kp_text = " ".join(model_output.key_points)
    completeness = _token_overlap(model_kp_text, gold_kp_text)

    overall = (
        accuracy * weights.get("accuracy_weight", 0.3)
        + structure * weights.get("structure_weight", 0.2)
        + faithfulness * weights.get("faithfulness_weight", 0.3)
        + completeness * weights.get("completeness_weight", 0.2)
    )

    return ScoreResult(
        accuracy=accuracy,
        structure=structure,
        faithfulness=faithfulness,
        completeness=completeness,
        overall=overall,
    )


def score_extraction(
    model_output: BaseModel,
    example: Dict[str, Any],
    gold: Dict[str, Any],
    weights: Dict[str, float],
) -> ScoreResult:
    """Simple extraction scoring.

    - accuracy: average Jaccard overlap over list fields + exact match for scalar fields when present.
    - structure: 1 if schema validated.
    - faithfulness: penalize if extracted medications/conditions are not substrings of the source.
    - completeness: fraction of gold list elements that appear in the prediction.
    """
    source = example["input"]

    score_components = []
    completeness_components = []

    # Scalars
    for field in ["name", "age", "location"]:
        gold_val = gold.get(field)
        pred_val = getattr(model_output, field)
        if gold_val is None and pred_val is None:
            score_components.append(1.0)
            completeness_components.append(1.0)
        elif gold_val is None or pred_val is None:
            score_components.append(0.0)
            completeness_components.append(0.0)
        else:
            score_components.append(1.0 if gold_val == pred_val else 0.0)
            completeness_components.append(1.0 if gold_val == pred_val else 0.0)

    # Lists
    for field in ["conditions", "medications"]:
        gold_list = set(gold.get(field, []))
        pred_list = set(getattr(model_output, field))

        if not gold_list and not pred_list:
            jaccard = 1.0
            completeness_f = 1.0
        else:
            intersection = gold_list & pred_list
            union = gold_list | pred_list
            jaccard = len(intersection) / len(union) if union else 0.0
            completeness_f = len(intersection) / len(gold_list) if gold_list else 0.0

        score_components.append(jaccard)
        completeness_components.append(completeness_f)

    accuracy = sum(score_components) / len(score_components)
    completeness = sum(completeness_components) / len(completeness_components)

    structure = 1.0

    # naive faithfulness: check that all predicted conditions/medications appear in the source
    ungrounded = 0
    total_items = 0
    for field in ["conditions", "medications"]:
        for item in getattr(model_output, field):
            total_items += 1
            if item.lower() not in source.lower():
                ungrounded += 1
    faithfulness = 1.0 if total_items == 0 else max(0.0, 1.0 - ungrounded / total_items)

    overall = (
        accuracy * weights.get("accuracy_weight", 0.3)
        + structure * weights.get("structure_weight", 0.3)
        + faithfulness * weights.get("faithfulness_weight", 0.2)
        + completeness * weights.get("completeness_weight", 0.2)
    )

    return ScoreResult(
        accuracy=accuracy,
        structure=structure,
        faithfulness=faithfulness,
        completeness=completeness,
        overall=overall,
    )


def score_example(
    task_name: str,
    schema_cls: type[BaseModel],
    output_text: str,
    example: Dict[str, Any],
    gold: Dict[str, Any],
    weights: Dict[str, float],
) -> ScoreResult:
    """High-level scoring entry point used by the eval runner."""
    try:
        parsed = parse_with_schema(output_text, schema_cls)
    except (ValidationError, json.JSONDecodeError) as e:
        # Total structure failure
        return ScoreResult(
            accuracy=0.0,
            structure=0.0,
            faithfulness=0.0,
            completeness=0.0,
            overall=0.0,
            parse_error=str(e),
        )

    if task_name == "classification":
        return score_classification(parsed, gold, weights)
    elif task_name == "summarization":
        return score_summarization(parsed, example, gold, weights)
    elif task_name == "extraction":
        return score_extraction(parsed, example, gold, weights)
    else:
        raise ValueError(f"Unknown task: {task_name}")
