from scripts import run_eval

def test_module_has_compute_accuracy():
    assert hasattr(run_eval, "compute_accuracy")
