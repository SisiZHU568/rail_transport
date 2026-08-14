"""测试阶段 A 配置边界和只读配置对象。"""

from copy import deepcopy

import pytest

from src.config import load_config, validate_config
from src.orchestration_config import load_phase_a_config


def test_debug_config_has_no_onboard_compute_fields() -> None:
    """列车仍是通信实体，但不能再作为 VNF 计算候选。"""

    config = load_config("configs/debug.yaml")

    assert "include_onboard" not in config["topology"]
    assert all(
        not key.startswith("onboard_")
        for key in config["node_resources"]
    )


def test_phase_a_config_uses_ctmc_rates_and_pair_ranges() -> None:
    """节点规模、实例数和保留档位必须全部由配置生成。"""

    model = load_phase_a_config(load_config("configs/debug.yaml"))

    assert model.fast_slot_seconds == pytest.approx(1.0)
    assert model.slow_frame_slots == 10
    assert model.failure_base_seed == 42
    assert set(model.domain_rates) == {0, 1, 2, 3}
    assert set(model.node_rates) == {0, 1, 2, 3, 4, 5}
    assert model.deployment_pairs[(0, 0)].cold_start_seconds == pytest.approx(0.3)
    assert model.allowed_instance_counts(0, 0) == (0, 1, 2, 3, 4)
    assert model.allowed_instance_counts(0, 5) == (0, 1, 2, 3, 4)
    assert model.retention_slot_options == (0, 5, 10, 15, 20)


@pytest.mark.parametrize(
    ("failure_rate", "recovery_rate"),
    ((-0.1, 1.0), (1.0, -0.1), (0.0, 0.0)),
)
def test_phase_a_config_rejects_invalid_ctmc_rates(
    failure_rate: float,
    recovery_rate: float,
) -> None:
    """连续时间率必须非负，且不能同时为零。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["runtime_failure"]["domain_rates"][0].update(
        failure_rate_per_second=failure_rate,
        recovery_rate_per_second=recovery_rate,
    )

    with pytest.raises(ValueError, match="domain_rates"):
        validate_config(config)


def test_phase_a_config_rejects_unknown_deployment_node() -> None:
    """允许部署组合引用未知节点时必须在启动前拒绝。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["instance_lifecycle"]["allowed_deployments"][0]["node_ids"] = [99]

    with pytest.raises(ValueError, match="allowed_deployments.*node"):
        validate_config(config)


def test_phase_a_config_rejects_cold_start_unit_mismatch() -> None:
    """VNF 毫秒字段和部署组合秒字段必须表达同一个冷启动时延。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["instance_lifecycle"]["allowed_deployments"][0][
        "cold_start_seconds"
    ] = 9.0

    with pytest.raises(ValueError, match="cold_start_seconds"):
        validate_config(config)


def test_phase_a_config_mappings_are_immutable() -> None:
    """模块获得的只读模型不能被运行代码原地篡改。"""

    model = load_phase_a_config(load_config("configs/debug.yaml"))

    with pytest.raises(TypeError):
        model.node_rates[0] = model.node_rates[1]  # type: ignore[index]
