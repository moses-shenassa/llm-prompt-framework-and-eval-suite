import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
from jinja2 import Template


MARKDOWN_TEMPLATE = """
# Evaluation Report – {{ run_name }}

- Date: {{ date }}
- Total examples: {{ total_examples }}
- Average overall score: {{ avg_overall | round(3) }}

## Scores by Task

{% for task_name, stats in by_task.items() %}
### Task: {{ task_name }}

- Examples: {{ stats.count }}
- Avg overall: {{ stats.avg_overall | round(3) }}
- Avg accuracy: {{ stats.avg_accuracy | round(3) }}
- Avg structure: {{ stats.avg_structure | round(3) }}
- Avg faithfulness: {{ stats.avg_faithfulness | round(3) }}
- Avg completeness: {{ stats.avg_completeness | round(3) }}

{% endfor %}
"""


def _aggregate_by_task(df: pd.DataFrame) -> Dict[str, Dict]:
    grouped = df.groupby("task_name")
    result: Dict[str, Dict] = {}
    for task_name, group in grouped:
        result[task_name] = {
            "count": int(group.shape[0]),
            "avg_overall": float(group["overall"].mean()),
            "avg_accuracy": float(group["accuracy"].mean()),
            "avg_structure": float(group["structure"].mean()),
            "avg_faithfulness": float(group["faithfulness"].mean()),
            "avg_completeness": float(group["completeness"].mean()),
        }
    return result


def write_reports(
    rows: List[Dict],
    reports_dir: str,
    run_name_prefix: str,
) -> None:
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_name_prefix}_{timestamp}"

    df = pd.DataFrame(rows)
    csv_path = os.path.join(reports_dir, f"{run_name}.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate stats
    total_examples = df.shape[0]
    avg_overall = float(df["overall"].mean()) if total_examples else 0.0
    by_task = _aggregate_by_task(df)

    context = {
        "run_name": run_name,
        "date": datetime.utcnow().isoformat(),
        "total_examples": total_examples,
        "avg_overall": avg_overall,
        "by_task": by_task,
    }

    # Markdown report
    md_template = Template(MARKDOWN_TEMPLATE)
    md_content = md_template.render(**context)
    md_path = os.path.join(reports_dir, f"{run_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Very simple HTML wrapping of markdown (could be replaced with a real renderer)
    html_path = os.path.join(reports_dir, f"{run_name}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body><pre>")
        f.write(md_content)
        f.write("</pre></body></html>")
