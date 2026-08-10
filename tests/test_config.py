"""测试 YAML 配置读取及 DPPO 必需配置的结构校验。"""

from copy import deepcopy

import pytest

from src.config import load_config, validate_config


def test_load_debug_config() -> None:
    """
    读取 debug.yaml 并检查几个关键参数。
    """

    config = load_config("configs/debug.yaml")

    assert config["simulation"]["random_seed"] == 42
    assert config["topology"]["mec_count"] == 5
    assert config["serverless"]["survival_slots"] == 20
    assert len(config["rl_scenario"]["functions"]) == 3
    assert config["rl_scenario"]["sfc"]["function_ids"] == [0, 1, 2]
    assert config["dppo"]["action"]["maximum_retention_seconds"] == 20.0
    assert config["dppo"]["diffusion"]["steps"] == 20
    assert config["dppo"]["diffusion"]["fine_tuned_steps"] == 5


def test_validate_config_rejects_dataset_fractions_not_summing_to_one() -> None:
    """数据切分比例错误时应在训练开始前给出明确提示。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["train_fraction"] = 0.8

    with pytest.raises(ValueError, match="数据集切分比例之和必须为 1"):
        validate_config(config)


def test_validate_config_requires_cloud_for_current_dppo_schema() -> None:
    """首版平铺状态包含中心云，因此关闭中心云必须被明确拒绝。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["topology"]["include_cloud"] = False

    with pytest.raises(ValueError, match="include_cloud=true"):
        validate_config(config)


def test_validate_config_matches_fault_domains_to_mec_count() -> None:
    """每个 MEC 都必须有一个显式故障域，扩容时不能沿用隐藏公式。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["topology"]["mec_fault_domain_ids"] = [0, 1]

    with pytest.raises(ValueError, match="mec_fault_domain_ids.*mec_count"):
        validate_config(config)


def test_validate_config_rejects_unencodable_replica_threshold() -> None:
    """阈值为 -1 时没有更小的合法动作值可用于表示 2 副本。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"]["replica_threshold"] = -1.0

    with pytest.raises(ValueError, match=r"replica_threshold.*\(-1, 1\]"):
        validate_config(config)
