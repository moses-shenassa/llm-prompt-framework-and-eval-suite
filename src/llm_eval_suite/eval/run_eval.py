import argparse
import importlib
import os
from typing import Any, Dict, List

import yaml

from llm_eval_suite.eval.loaders import align_examples_and_gold, load_jsonl
from llm_eval_suite.eval.reporting import write_reports
from llm_eval_suite.eval.scoring import score_example
from llm_eval_suite.eval.safety_checks import run_safety_checks
from llm_eval_suite.utils.llm_client import LLMClient
from llm_eval_suite.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(schema_module: str, schema_class: str):
    module = importlib.import_module(schema_module)
    return getattr(module, schema_class)


def build_user_prompt(prompt_template_path: str, variables: Dict[str, Any]) -> str:
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read()
    result = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        if isinstance(value, list):
            # For allowed_labels we may want a comma-separated string
            value_str = ", ".join(map(str, value))
        else:
            value_str = str(value)
        result = result.replace(placeholder, value_str)
    return result


def main(config_path: str) -> None:
    config = load_config(config_path)
    model_cfg = config["model"]
    output_cfg = config["output"]

    client = LLMClient(
        provider=model_cfg["provider"],
        model_name=model_cfg["name"],
        temperature=float(model_cfg.get("temperature", 0.0)),
    )

    all_rows: List[Dict[str, Any]] = []

    for task in config["tasks"]:
        if not task.get("enabled", True):
            continue

        task_name = task["name"]
        logger.info("Running task: %s", task_name)

        dataset = load_jsonl(task["dataset_path"])
        gold = load_jsonl(task["gold_path"])

        schema_cls = load_schema(task["schema_module"], task["schema_class"])
        weights = task.get("scoring", {})

        few_shots_path = task.get("few_shots_path")
        few_shots: List[Dict[str, Any]] = []
        if few_shots_path and os.path.exists(few_shots_path):
            import json

            with open(few_shots_path, "r", encoding="utf-8") as f:
                few_shots = json.load(f)

        for example, gold_ex in align_examples_and_gold(dataset, gold):
            ex_id = example["id"]
            logger.info("Evaluating %s example id=%s", task_name, ex_id)

            variables: Dict[str, Any] = {"input_text": example["input"]}
            if task_name == "classification":
                variables["allowed_labels"] = example.get("allowed_labels", gold_ex.get("allowed_labels", []))

            user_prompt = build_user_prompt(task["prompt_template_path"], variables)

            # Here we treat the entire template as the "user" prompt and use a generic system prompt.
            system_prompt = f"You are a helpful assistant performing the '{task_name}' task."

            output_text = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

            # Scoring
            score = score_example(
                task_name=task_name,
                schema_cls=schema_cls,
                output_text=output_text,
                example=example,
                gold=gold_ex,
                weights=weights,
            )

            combined_text = f"USER INPUT:\n{example['input']}\n\nMODEL OUTPUT:\n{llm_output}"
	    safety = run_safety_checks(combined_text)


            row = {
                "task_name": task_name,
                "example_id": ex_id,
                "output_text": output_text,
                "accuracy": score.accuracy,
                "structure": score.structure,
                "faithfulness": score.faithfulness,
                "completeness": score.completeness,
                "overall": score.overall,
                "parse_error": score.parse_error,
                "safety_violation": safety.has_violation,
                "safety_reasons": "; ".join(safety.reasons),
            }
            all_rows.append(row)

    write_reports(
        rows=all_rows,
        reports_dir=output_cfg["reports_dir"],
        run_name_prefix=output_cfg.get("run_name_prefix", "eval_run"),
    )
    logger.info("Evaluation complete. Wrote %d rows.", len(all_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation harness.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/llm_eval_suite/config/default_config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
