from pathlib import Path

from neuro_validate.src.config import load_benchmark_config


def test_load_benchmark_config_parses_simple_yaml() -> None:
    config = load_benchmark_config(
        Path("/Users/anish/PERSONAL/emotiv_learn/neuro_validate/configs/cog_bci_nback_workload.yaml")
    )

    assert config.benchmark_name == "cog_bci_nback_workload"
    assert config.task_name == "N-Back"
    assert config.frequency_bands["theta"] == (4.0, 8.0)
    assert "balanced_accuracy" in config.metrics

