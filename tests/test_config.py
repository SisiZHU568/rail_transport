"""
test_config.py

测试 YAML 配置文件能否被正确读取。
"""

from src.config import load_config


def test_load_debug_config() -> None:
    """
    读取 debug.yaml 并检查几个关键参数。
    """

    config = load_config("configs/debug.yaml")

    assert config["simulation"]["random_seed"] == 42
    assert config["topology"]["mec_count"] == 5
    assert config["serverless"]["survival_slots"] == 20