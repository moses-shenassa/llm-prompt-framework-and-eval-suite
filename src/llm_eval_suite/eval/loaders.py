import json
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    examples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def align_examples_and_gold(
    dataset: List[Dict],
    gold: List[Dict],
    id_key: str = "id",
) -> Iterable[Tuple[Dict, Dict]]:
    """Yield (example, gold) pairs aligned by an ID field.

    Assumes both lists contain unique IDs on the given key.
    """
    gold_by_id: Dict[str, Dict] = {g[id_key]: g for g in gold}
    for ex in dataset:
        ex_id = ex[id_key]
        if ex_id not in gold_by_id:
            raise ValueError(f"No gold label found for id={ex_id!r}")
        yield ex, gold_by_id[ex_id]
