from llm_eval_suite.eval.safety_checks import run_safety_checks


def test_safety_detects_forbidden_phrase():
    out = run_safety_checks("You should kill yourself.")
    assert out.has_violation
    assert any("kill yourself" in reason for reason in out.reasons)
