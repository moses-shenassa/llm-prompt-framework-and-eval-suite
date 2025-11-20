import pytest

from llm_eval_suite.utils.llm_client import LLMClient


def test_llm_client_invalid_provider():
    with pytest.raises(ValueError):
        LLMClient(provider="unknown", model_name="test-model")
