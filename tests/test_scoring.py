from llm_eval_suite.eval.scoring import _token_overlap


def test_token_overlap_basic():
    a = "the quick brown fox"
    b = "quick brown fox jumps"
    score = _token_overlap(a, b)
    assert 0 < score < 1
