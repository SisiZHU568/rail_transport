"""
config.py

本文件负责读取 YAML 配置文件。

例如：
    configs/debug.yaml

读取后，YAML 内容会被转换成 Python 字典。
"""

import math
from pathlib import Path
from typing import Any

import yaml

from src.orchestration_config import load_phase_a_config


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """读取必需的字典配置，并给出比普通 ``KeyError`` 更清楚的提示。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是字典。")
    return value


def _require_positive_integer(
    parent: dict[str, Any],
    key: str,
    *,
    display_key: str | None = None,
) -> int:
    """读取严格大于零的整数配置，布尔值不能冒充整数。"""

    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"配置项 {display_key or key} 必须是正整数。")
    return value


def validate_config(config: dict[str, Any]) -> None:
    """校验首版 DPPO 运行前必须确定的配置结构和取值范围。

    此函数只检查跨模块都会依赖的边界。VNF、SFC 等实体的详细物理参数，
    继续由对应数据类和 ``rl_scenario`` 构造器负责检查，避免重复实现规则。
    """

    topology = _require_mapping(config, "topology")
    if topology.get("include_cloud") is not True:
        raise ValueError("DPPO 平铺状态要求 topology.include_cloud=true。")
    mec_count = _require_positive_integer(topology, "mec_count")
    if mec_count < 2:
        raise ValueError("支持三副本时 topology.mec_count 至少为 2。")
    fault_domain_ids = topology.get("mec_fault_domain_ids")
    if (
        not isinstance(fault_domain_ids, list)
        or len(fault_domain_ids) != mec_count
        or any(
            isinstance(domain_id, bool)
            or not isinstance(domain_id, int)
            or domain_id < 0
            for domain_id in fault_domain_ids
        )
    ):
        raise ValueError(
            "topology.mec_fault_domain_ids 必须包含与 mec_count "
            "相同数量的非负整数。"
        )

    scenario = _require_mapping(config, "rl_scenario")
    functions = scenario.get("functions")
    if not isinstance(functions, list) or not functions:
        raise ValueError("配置项 rl_scenario.functions 必须是非空列表。")
    if any(not isinstance(item, dict) for item in functions):
        raise ValueError("rl_scenario.functions 中每个 VNF 必须是字典。")
    _require_mapping(scenario, "sfc")

    dppo = _require_mapping(config, "dppo")
    action = _require_mapping(dppo, "action")
    maximum_retention = action.get("maximum_retention_seconds")
    if (
        isinstance(maximum_retention, bool)
        or not isinstance(maximum_retention, (int, float))
        or not math.isfinite(float(maximum_retention))
        or float(maximum_retention) <= 0.0
    ):
        raise ValueError("maximum_retention_seconds 必须是正有限数。")
    minimum_replicas = _require_positive_integer(action, "minimum_replicas")
    maximum_replicas = _require_positive_integer(action, "maximum_replicas")
    if minimum_replicas > maximum_replicas:
        raise ValueError("minimum_replicas 不能大于 maximum_replicas。")
    # DPPO 当前只把轨旁 MEC 和中心云作为 VNF 部署节点，车载节点不计入上限。
    compute_node_count = mec_count + int(topology.get("include_cloud") is True)
    if maximum_replicas > compute_node_count:
        raise ValueError("maximum_replicas 不能大于 compute_node_count。")

    fast_scheduler = _require_mapping(dppo, "fast_scheduler")
    if fast_scheduler.get("solver") != "CLARABEL":
        raise ValueError("dppo.fast_scheduler.solver 当前必须为 CLARABEL。")
    _require_positive_integer(
        fast_scheduler,
        "max_iterations",
        display_key="dppo.fast_scheduler.max_iterations",
    )
    feasibility_tolerance = fast_scheduler.get("feasibility_tolerance")
    if (
        isinstance(feasibility_tolerance, bool)
        or not isinstance(feasibility_tolerance, (int, float))
        or not math.isfinite(float(feasibility_tolerance))
        or float(feasibility_tolerance) <= 0.0
    ):
        raise ValueError(
            "dppo.fast_scheduler.feasibility_tolerance 必须是正有限数。"
        )

    diffusion = _require_mapping(dppo, "diffusion")
    diffusion_steps = _require_positive_integer(diffusion, "steps")
    fine_tuned_steps = _require_positive_integer(diffusion, "fine_tuned_steps")
    if fine_tuned_steps > diffusion_steps:
        raise ValueError("fine_tuned_steps 不能大于 diffusion.steps。")

    dataset = _require_mapping(dppo, "dataset")
    fractions = tuple(
        dataset.get(key)
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 < float(value) < 1.0
        for value in fractions
    ):
        raise ValueError("每个数据集切分比例必须位于 (0, 1)。")
    if not math.isclose(sum(float(value) for value in fractions), 1.0):
        raise ValueError("数据集切分比例之和必须为 1。")
    _require_positive_integer(dataset, "episodes")
    teacher_schema_version = dataset.get("teacher_schema_version")
    if (
        not isinstance(teacher_schema_version, str)
        or not teacher_schema_version.strip()
    ):
        raise ValueError(
            "dppo.dataset.teacher_schema_version 必须是非空字符串。"
        )
    teacher_names = dataset.get("teacher_names")
    allowed_teachers = {"cost", "reliability", "balanced"}
    if (
        not isinstance(teacher_names, list)
        or not teacher_names
        or any(
            not isinstance(name, str) or name not in allowed_teachers
            for name in teacher_names
        )
        or len(teacher_names) != len(set(teacher_names))
    ):
        raise ValueError(
            "dppo.dataset.teacher_names 必须是由 cost、reliability、balanced "
            "组成的非空无重复列表。"
        )
    _require_positive_integer(dataset, "max_slow_steps_per_episode")
    seed_start = dataset.get("seed_start")
    split_seed = dataset.get("split_seed")
    for key, value in (("seed_start", seed_start), ("split_seed", split_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"dppo.dataset.{key} 必须是非负整数。")

    pretraining = _require_mapping(dppo, "pretraining")
    _require_positive_integer(
        pretraining,
        "optimizer_steps",
        display_key="dppo.pretraining.optimizer_steps",
    )
    _require_positive_integer(
        pretraining,
        "validation_interval_steps",
        display_key="dppo.pretraining.validation_interval_steps",
    )
    _require_positive_integer(
        pretraining,
        "batch_size",
        display_key="dppo.pretraining.batch_size",
    )
    for key in ("learning_rate", "gradient_clip_norm"):
        value = pretraining.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"dppo.pretraining.{key} 必须是正有限数。")
    pretraining_seed = pretraining.get("seed")
    if (
        isinstance(pretraining_seed, bool)
        or not isinstance(pretraining_seed, int)
        or pretraining_seed < 0
    ):
        raise ValueError("dppo.pretraining.seed 必须是非负整数。")
    pretraining_output = pretraining.get("output_root")
    if not isinstance(pretraining_output, str) or not pretraining_output:
        raise ValueError("dppo.pretraining.output_root 必须是非空字符串。")

    training = _require_mapping(dppo, "training")
    state_schema_version = training.get("state_schema_version")
    if not isinstance(state_schema_version, str) or not state_schema_version:
        raise ValueError("dppo.training.state_schema_version 必须是非空字符串。")
    for key in (
        "iterations",
        "episodes_per_iteration",
        "batch_size",
        "update_epochs",
    ):
        _require_positive_integer(
            training,
            key,
            display_key=f"dppo.training.{key}",
        )
    bounded_values = {
        "gamma": (0.0, 1.0, False),
        "gae_lambda": (0.0, 1.0, True),
        "clip_ratio": (0.0, 1.0, False),
        "denoising_discount": (0.0, 1.0, False),
    }
    for key, (minimum, maximum, allow_zero) in bounded_values.items():
        value = training.get(key)
        valid_number = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
        lower_valid = (
            float(value) >= minimum if valid_number and allow_zero
            else valid_number and float(value) > minimum
        )
        if not lower_valid or float(value) > maximum:
            left_bracket = "[" if allow_zero else "("
            raise ValueError(
                f"dppo.training.{key} 必须位于 {left_bracket}0, 1]。"
            )
    for key in (
        "policy_learning_rate",
        "value_learning_rate",
        "gradient_clip_norm",
    ):
        value = training.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"dppo.training.{key} 必须是正有限数。")
    hidden_dims = training.get("value_hidden_dims")
    if (
        not isinstance(hidden_dims, list)
        or not hidden_dims
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in hidden_dims
        )
    ):
        raise ValueError("dppo.training.value_hidden_dims 必须是非空正整数列表。")
    training_seed = training.get("seed")
    if (
        isinstance(training_seed, bool)
        or not isinstance(training_seed, int)
        or training_seed < 0
    ):
        raise ValueError("dppo.training.seed 必须是非负整数。")
    if training.get("device") not in {"cpu", "cuda"}:
        raise ValueError("dppo.training.device 必须是 cpu 或 cuda。")
    training_output = training.get("output_root")
    if not isinstance(training_output, str) or not training_output:
        raise ValueError("dppo.training.output_root 必须是非空字符串。")

    # 阶段 A 的物理单位、CTMC 率和 VNF—节点组合在同一次加载中校验，
    # 避免仿真运行到一半才发现引用或容量配置错误。
    load_phase_a_config(config)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    读取 YAML 配置文件。

    Parameters
    ----------
    config_path:
        配置文件路径，例如 "configs/debug.yaml"。

    Returns
    -------
    dict[str, Any]:
        读取后的配置字典。

    Raises
    ------
    FileNotFoundError:
        当配置文件不存在时抛出。

    ValueError:
        当配置文件为空，或者最外层不是字典时抛出。
    """

    # 将字符串路径转换成 Path 对象，
    # 便于后续检查文件是否存在和打开文件。
    path = Path(config_path)

    # 检查配置文件是否存在。
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    # 使用 UTF-8 编码打开文件，避免中文乱码。
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 空 YAML 文件读取后会得到 None。
    if config is None:
        raise ValueError(f"配置文件为空：{path}")

    # 本项目要求 YAML 最外层必须是键值对结构。
    if not isinstance(config, dict):
        raise ValueError("配置文件最外层必须是字典结构。")

    # 配置加载后立即验证，使错误在训练或仿真开始前暴露。
    validate_config(config)

    return config
