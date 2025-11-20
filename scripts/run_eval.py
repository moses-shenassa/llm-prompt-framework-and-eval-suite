"""
scripts/run_eval.py

Simple end-to-end runner for a single evaluation task, designed to be easy for
recruiters and interviewers to understand. It does not depend on the rest of the
package internals – it is a minimal, self-contained example runner.

Usage:

    python scripts/run_eval.py --config configs/support_ticket_classification.yaml
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class ModelConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int


@dataclass
class TaskConfig:
    name: str
    type: str
    dataset_path: str
    input_field: str
    label_field: str
    prompt_template_name: str
    output_schema: str
    metrics: List[str]


@dataclass
class OutputConfig:
    output_dir: str
    run_name: str


PROMPT_TEMPLATES: Dict[str, str] = {
    "support_ticket_classification": (
        "You are an AI assistant helping a customer support team.\n"
        "Read the following support ticket and classify it into one of these categories:\n\n"
        "  - billing\n"
        "  - technical_issue\n"
        "  - account_access\n"
        "  - account_change\n"
        "  - product_feedback\n"
        "  - security\n\n"
        "Return a JSON object with exactly this structure:\n\n"
        "{{\"label\": \"<one_of_the_categories_above>\", \"reason\": \"<short explanation>\"}}\n\n"
        "Ticket:\n"
        "\"{text}\""
    )
}


def load_config(path: str) -> (ModelConfig, TaskConfig, OutputConfig):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = ModelConfig(
        provider=config["model"]["provider"],
        model=config["model"]["model"],
        temperature=float(config["model"].get("temperature", 0.0)),
        max_tokens=int(config["model"].get("max_tokens", 512)),
    )

    task_cfg = TaskConfig(
        name=config["task"]["name"],
        type=config["task"]["type"],
        dataset_path=config["task"]["dataset_path"],
        input_field=config["task"]["input_field"],
        label_field=config["task"]["label_field"],
        prompt_template_name=config["task"]["prompt_template_name"],
        output_schema=config["task"]["output_schema"],
        metrics=config["task"]["metrics"],
    )

    output_cfg = OutputConfig(
        output_dir=config["output"]["output_dir"],
        run_name=config["output"].get("run_name", "run"),
    )

    return model_cfg, task_cfg, output_cfg


def build_client(model_cfg: ModelConfig):
    if model_cfg.provider != "openai":
        raise ValueError(f"Only provider 'openai' is supported in this example, got: {model_cfg.provider}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment (.env).")
    client = OpenAI()
    return client


def call_model(client, model_cfg: ModelConfig, prompt: str) -> str:
    completion = client.responses.create(
        model=model_cfg.model,
        input=prompt,
        max_output_tokens=model_cfg.max_tokens,
        temperature=model_cfg.temperature,
    )
    return completion.output[0].content[0].text


def compute_accuracy(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for r in rows if r.get("gold_label") == r.get("model_label"))
    return correct / len(rows)


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple evaluation task.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    model_cfg, task_cfg, output_cfg = load_config(args.config)

    dataset_path = Path(task_cfg.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    import pandas as pd
    df = pd.read_csv(dataset_path)
    input_col = task_cfg.input_field
    label_col = task_cfg.label_field

    if input_col not in df.columns:
        raise ValueError(f"Input column '{input_col}' not found in dataset.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    prompt_template = PROMPT_TEMPLATES.get(task_cfg.prompt_template_name)
    if not prompt_template:
        raise ValueError(f"Prompt template '{task_cfg.prompt_template_name}' not found.")

    client = build_client(model_cfg)

    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        text = row[input_col]
        gold_label = row[label_col]

        prompt = prompt_template.format(text=text)
        raw_output = call_model(client, model_cfg, prompt)

        model_label = None
        structure_valid = False

        try:
            parsed = json.loads(raw_output)
            model_label = parsed.get("label")
            structure_valid = isinstance(model_label, str)
        except json.JSONDecodeError:
            parsed = None
            structure_valid = False

        results.append(
            {
                "id": row.get("id"),
                "text": text,
                "gold_label": gold_label,
                "raw_output": raw_output,
                "parsed_output": json.dumps(parsed) if parsed is not None else "",
                "model_label": model_label,
                "structure_valid": structure_valid,
            }
        )

    accuracy = compute_accuracy(results)
    struct_valid_rate = sum(1 for r in results if r["structure_valid"]) / len(results) if results else 0.0

    output_base = Path(output_cfg.output_dir) / output_cfg.run_name
    ensure_output_dir(output_base.parent)

    csv_path = output_base.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    md_path = output_base.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Summary – {task_cfg.name}\n\n")
        f.write(f"- Task type: {task_cfg.type}\n")
        f.write(f"- Dataset: {task_cfg.dataset_path}\n")
        f.write(f"- Model: {model_cfg.model} ({model_cfg.provider})\n")
        f.write(f"- Examples: {len(results)}\n\n")
        f.write("## Metrics\n\n")
        f.write(f"- Accuracy: {accuracy:.3f}\n")
        f.write(f"- Structure validity: {struct_valid_rate:.3f}\n")

    html_path = output_base.with_suffix(".html")
    with open(md_path, "r", encoding="utf-8") as f_md, open(html_path, "w", encoding="utf-8") as f_html:
        md_content = f_md.read()
        f_html.write("<html><body><pre>")
        f_html.write(md_content)
        f_html.write("</pre></body></html>")

    print(f"Wrote CSV results to: {csv_path}")
    print(f"Wrote Markdown summary to: {md_path}")
    print(f"Wrote HTML summary to: {html_path}")


if __name__ == "__main__":
    main()
