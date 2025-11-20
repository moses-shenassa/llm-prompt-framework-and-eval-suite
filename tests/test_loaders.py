from llm_eval_suite.eval.loaders import load_jsonl, align_examples_and_gold


def test_load_jsonl_and_alignment():
    dataset = load_jsonl("data/datasets/classification_examples.jsonl")
    gold = load_jsonl("data/gold_labels/classification_gold.jsonl")

    assert len(dataset) == len(gold) > 0

    # Ensure we can align without error and that IDs match
    for ex, gold_ex in align_examples_and_gold(dataset, gold):
        assert ex["id"] == gold_ex["id"]
