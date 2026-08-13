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
    assert config["dppo"]["action"]["schema_version"] == "joint-sfc-continuous-v2"
    assert config["dppo"]["action"]["minimum_replicas"] == 2
    assert config["dppo"]["action"]["maximum_replicas"] == 3
    assert config["dppo"]["fast_scheduler"] == {
        "solver": "CLARABEL",
        "max_iterations": 200,
        "feasibility_tolerance": 1.0e-7,
    }
    assert config["dppo"]["diffusion"]["steps"] == 20
    assert config["dppo"]["diffusion"]["fine_tuned_steps"] == 5
    assert config["dppo"]["pretraining"]["optimizer_steps"] == 200
    assert config["dppo"]["pretraining"]["validation_interval_steps"] == 10
    assert (
        config["dppo"]["dataset"]["teacher_schema_version"]
        == "balanced-min-replica-reliability-v1"
    )
    assert config["dppo"]["training"]["iterations"] > 0
    assert config["dppo"]["training"]["episodes_per_iteration"] > 0
    assert config["dppo"]["training"]["value_hidden_dims"] == [256, 256]
    assert config["dppo"]["training"]["state_schema_version"] == "dppo-v2-flat"
    assert config["dppo"]["training"]["normalize_advantages"] is True
    assert config["dppo"]["training"]["policy_learning_rate"] == 0.0001
    assert config["dppo"]["training"]["clip_ratio_base"] == 0.001
    stability = config["dppo"]["stability"]
    assert stability == {
        "training_sampling_min_std": 0.01,
        "probability_min_std": 0.10,
        "evaluation_sampling_min_std": 0.001,
        "target_kl": 1.0,
        "target_clip_fraction_min": 0.60,
        "target_clip_fraction_max": 0.85,
        "clip_ratio_candidates": [0.10, 0.01, 0.001],
        "calibration_iterations": 1,
        "calibration_episodes_per_iteration": 1,
        "calibration_seed_start": 20000,
    }
    assert config["dppo"]["training"]["seed"] == 13000
    assert 20000 <= stability["calibration_seed_start"] < 30000


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    (
        ("solver", "HIGHS"),
        ("max_iterations", 0),
        ("feasibility_tolerance", 0.0),
    ),
)
def test_validate_config_rejects_invalid_fast_scheduler_setting(
    key: str,
    invalid_value: object,
) -> None:
    """快层只允许使用一组明确且有物理意义的 CLARABEL 配置。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["fast_scheduler"][key] = invalid_value

    with pytest.raises(ValueError, match=rf"dppo\.fast_scheduler\.{key}"):
        validate_config(config)


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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_replicas", True),
        ("minimum_replicas", 0),
        ("maximum_replicas", 0),
        ("maximum_replicas", 2.5),
    ),
)
def test_validate_config_rejects_invalid_replica_bounds(
    field: str,
    value: object,
) -> None:
    """副本上下限必须是正整数，避免产生无法执行的部署数量。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"][field] = value

    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_validate_config_rejects_reversed_replica_bounds() -> None:
    """下限不能大于上限。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"]["minimum_replicas"] = 4
    config["dppo"]["action"]["maximum_replicas"] = 3

    with pytest.raises(ValueError, match="minimum_replicas.*maximum_replicas"):
        validate_config(config)


def test_validate_config_rejects_replica_maximum_above_compute_nodes() -> None:
    """副本上限不能超过 DPPO 实际可部署的 MEC 与中心云节点总数。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["action"]["maximum_replicas"] = 99

    with pytest.raises(ValueError, match="compute_node_count"):
        validate_config(config)


def test_validate_config_requires_known_nonempty_dataset_teachers() -> None:
    """数据生成教师必须在配置中显式声明且名称可识别。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_names"] = ["unknown"]

    with pytest.raises(ValueError, match="teacher_names"):
        validate_config(config)


@pytest.mark.parametrize("invalid_value", ("", "   ", 1, None))
def test_validate_config_requires_nonempty_teacher_schema_version(
    invalid_value: object,
) -> None:
    """教师规则版本必须明确，避免新旧专家标签共享同一个配置哈希。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["teacher_schema_version"] = invalid_value

    with pytest.raises(ValueError, match="teacher_schema_version"):
        validate_config(config)


def test_validate_config_requires_teacher_schema_version_field() -> None:
    """缺少教师规则版本时必须立即拒绝，而不是生成无法追溯的数据。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"].pop("teacher_schema_version", None)

    with pytest.raises(ValueError, match="teacher_schema_version"):
        validate_config(config)


def test_validate_config_requires_positive_dataset_slow_steps() -> None:
    """每条 Episode 的采集上限不能写死，也不能配置为零。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["dataset"]["max_slow_steps_per_episode"] = 0

    with pytest.raises(ValueError, match="max_slow_steps_per_episode"):
        validate_config(config)


def test_validate_config_requires_positive_pretraining_batch_size() -> None:
    """扩散预训练参数错误时应在读取配置阶段暴露。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["pretraining"] = {
        "optimizer_steps": 1,
        "validation_interval_steps": 1,
        "batch_size": 0,
        "learning_rate": 0.0003,
        "gradient_clip_norm": 5.0,
        "seed": 12000,
        "output_root": "results/dppo/pretraining",
    }

    with pytest.raises(ValueError, match="pretraining.batch_size"):
        validate_config(config)


@pytest.mark.parametrize("key", ("optimizer_steps", "validation_interval_steps"))
def test_validate_config_requires_positive_pretraining_step_counts(key: str) -> None:
    """训练量和验证间隔都必须用正的优化器更新次数表达。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["pretraining"][key] = 0

    with pytest.raises(ValueError, match=rf"dppo\.pretraining\.{key}"):
        validate_config(config)


def test_validate_config_requires_positive_online_training_iterations() -> None:
    """在线训练轮数不能依赖脚本默认值，也不能配置为零。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["dppo"]["training"]["iterations"] = 0

    with pytest.raises(ValueError, match="dppo.training.iterations"):
        validate_config(config)
