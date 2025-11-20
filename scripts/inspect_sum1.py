import os
import json
import pandas as pd

from llm_eval_suite.eval.loaders import load_jsonl
from llm_eval_suite.eval.scoring import score_example
from llm_eval_suite.schemas.summarization_schema import SummarizationOutput

# 1. Load the example and gold for sum-1
examples = load_jsonl("data/datasets/summarization_examples.jsonl")
gold = load_jsonl("data/gold_labels/summarization_gold.jsonl")

example = next(e for e in examples if e["id"] == "sum-1")
gold_ex = next(g for g in gold if g["id"] == "sum-1")

print("=== INPUT (sum-1) ===")
print(example["input"])
print()

print("=== GOLD SUMMARY ===")
print(gold_ex["summary"])
print()
print("=== GOLD KEY POINTS ===")
print("\n".join(gold_ex["key_points"]))
print()

# 2. Find the model's output in the latest CSV report
reports = [f for f in os.listdir("reports") if f.endswith(".csv")]
if not reports:
    raise SystemExit("No CSV reports found in ./reports. Run the eval harness first.")

latest_csv = max(reports, key=lambda f: os.path.getmtime(os.path.join("reports", f)))
print(f"Using latest report: {latest_csv}")

df = pd.read_csv(os.path.join("reports", latest_csv))
row = df[(df["task_name"] == "summarization") & (df["example_id"] == "sum-1")].iloc[0]

print()
print("=== MODEL RAW OUTPUT TEXT ===")
print(row["output_text"])
print()

# 3. Parse model output and compute scores again (for illustration)
model_output = SummarizationOutput.model_validate(json.loads(row["output_text"]))

weights = {
    "accuracy_weight": 0.3,
    "structure_weight": 0.2,
    "faithfulness_weight": 0.3,
    "completeness_weight": 0.2,
}

score = score_example(
    task_name="summarization",
    schema_cls=SummarizationOutput,
    output_text=row["output_text"],
    example=example,
    gold=gold_ex,
    weights=weights,
)

print("=== SCORES ===")
print(f"Accuracy:     {score.accuracy:.3f}")
print(f"Structure:    {score.structure:.3f}")
print(f"Faithfulness: {score.faithfulness:.3f}")
print(f"Completeness: {score.completeness:.3f}")
print(f"OVERALL:      {score.overall:.3f}")
print(f"Parse error:  {score.parse_error}")
