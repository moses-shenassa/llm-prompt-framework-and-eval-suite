from scripts.run_eval import compute_accuracy

def test_compute_accuracy_trivial():
    rows = [
        {"gold_label": "a", "model_label": "a"},
        {"gold_label": "b", "model_label": "b"},
        {"gold_label": "c", "model_label": "x"},
        {"gold_label": "d", "model_label": "d"},
    ]
    acc = compute_accuracy(rows)
    assert abs(acc - 0.75) < 1e-9
