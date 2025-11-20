import yaml
from pathlib import Path

def test_config_loading_support_ticket():
    config_path = Path("configs/support_ticket_classification.yaml")
    assert config_path.exists(), "support_ticket_classification.yaml should exist"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg
    assert "task" in cfg
    assert "output" in cfg

    assert cfg["task"]["type"] == "classification"
    assert cfg["task"]["dataset_path"].endswith("support_tickets.csv")
