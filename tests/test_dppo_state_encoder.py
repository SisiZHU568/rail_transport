"""测试配置驱动的 DPPO 平铺状态编码器。"""

from dataclasses import replace

import numpy as np
import pytest

from src.dppo_state_encoder import (
    DPPO_HISTORY_FEATURE_NAMES,
    DPPO_STATE_SCHEMA_VERSION,
    DPPOFunctionObservation,
    DPPOHistoryEncoder,
    DPPOHistoryObservation,
    DPPONodeObservation,
    DPPOSFCObservation,
    DPPOStateEncoder,
    DPPOStateSnapshot,
)
from src.scenario_dimensions import ScenarioDimensions


def test_state_schema_version_is_explicit() -> None:
    """数据集和检查点必须共享同一状态版本，不能只凭相同维度判断兼容。"""

    assert DPPO_STATE_SCHEMA_VERSION == "dppo-v1-flat"


@pytest.mark.parametrize(
    ("mec_count", "function_count"),
    [(3, 2), (5, 3), (8, 3)],
)
def test_state_shape_is_derived_from_configuration(
    mec_count: int,
    function_count: int,
) -> None:
    """MEC 或 VNF 数量变化时，状态形状必须由统一维度对象自动计算。"""

    dimensions = ScenarioDimensions(
        tuple(range(mec_count)),
        mec_count,
        tuple(range(function_count)),
    )
    encoder = DPPOStateEncoder(dimensions)

    encoded = encoder.encode(
        DPPOStateSnapshot.zeros(dimensions),
        DPPOHistoryObservation.zeros(),
    )

    assert encoded.shape == (dimensions.state_dim,)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()
    assert ((0.0 <= encoded) & (encoded <= 1.0)).all()
    assert encoder.feature_names[-11:] == DPPO_HISTORY_FEATURE_NAMES


def test_history_features_have_locked_dppo_order() -> None:
    """最后 11 维使用 DPPO 动作与执行结果语义，不再编码旧离散动作。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0, 1))
    history = DPPOHistoryObservation(
        previous_mean_replica_count=2.5 / 3.0,
        previous_three_replica_ratio=0.25,
        previous_primary_retention_ratio=0.30,
        previous_backup_retention_ratio=0.20,
        previous_projection_change_ratio=0.10,
        current_hot_replica_ratio=0.50,
        current_cloud_replica_ratio=0.40,
        previous_success_rate=0.90,
        previous_sla_violation_rate=0.10,
        previous_normalized_cost=0.35,
        previous_repair_failure_rate=0.05,
    )

    encoded = DPPOStateEncoder(dimensions).encode(
        DPPOStateSnapshot.zeros(dimensions),
        history,
    )

    np.testing.assert_allclose(
        encoded[-11:],
        np.asarray(
            [
                2.5 / 3.0,
                0.25,
                0.30,
                0.20,
                0.10,
                0.50,
                0.40,
                0.90,
                0.10,
                0.35,
                0.05,
            ],
            dtype=np.float32,
        ),
    )
    assert DPPOHistoryEncoder().encode(history) == pytest.approx(
        encoded[-11:].tolist()
    )


def test_feature_names_and_values_follow_configured_ids() -> None:
    """非连续节点和 VNF 编号也必须按照 ScenarioDimensions 中的顺序编码。"""

    dimensions = ScenarioDimensions((10, 20, 30), 99, (7, 8))
    snapshot = DPPOStateSnapshot(
        serving_mec=20,
        next_mec=30,
        route_progress=0.25,
        normalized_remaining_dwell=0.50,
        normalized_mean_requests=0.40,
        normalized_peak_requests=0.60,
        normalized_load_trend=-0.50,
        global_failure_risk=0.10,
        node_observations=tuple(
            DPPONodeObservation(
                node_id=node_id,
                free_cpu_ratio=0.80,
                free_memory_ratio=0.70,
                base_availability=0.99,
                predicted_failure_probability=0.01,
                operational=node_id != 30,
                normalized_delay_from_serving=0.20,
            )
            for node_id in dimensions.compute_node_ids
        ),
        function_observations=(
            DPPOFunctionObservation(7, 0.20, 0.30, 0.40, 0.50),
            DPPOFunctionObservation(8, 0.25, 0.35, 0.45, 0.55),
        ),
        sfc_observation=DPPOSFCObservation(0.99, 0.75, 0.20),
    )
    encoder = DPPOStateEncoder(dimensions)

    encoded = encoder.encode(snapshot, DPPOHistoryObservation.zeros())

    assert encoder.feature_names[:3] == (
        "current_mec_10",
        "current_mec_20",
        "current_mec_30",
    )
    assert encoder.feature_names[3:6] == (
        "next_mec_10",
        "next_mec_20",
        "next_mec_30",
    )
    np.testing.assert_array_equal(
        encoded[:6],
        np.asarray([0, 1, 0, 0, 0, 1], dtype=np.float32),
    )
    assert encoded[6] == pytest.approx(0.25)
    # 负载趋势由 [-1, 1] 线性映射到神经网络统一使用的 [0, 1]。
    assert encoded[10] == pytest.approx(0.25)
    assert "node_99_free_cpu_ratio" in encoder.feature_names
    assert "function_7_normalized_cpu" in encoder.feature_names
    assert "function_8_normalized_cold_start_time" in encoder.feature_names


def test_zero_snapshot_has_no_fake_history() -> None:
    """Episode 首状态的历史 11 维必须全零，不能伪造默认动作结果。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    encoded = DPPOStateEncoder(dimensions).encode(
        DPPOStateSnapshot.zeros(dimensions),
        DPPOHistoryObservation.zeros(),
    )

    np.testing.assert_array_equal(encoded[-11:], np.zeros(11, dtype=np.float32))


@pytest.mark.parametrize("mutation", ["missing", "reordered", "duplicate"])
def test_node_observations_must_match_configured_order(mutation: str) -> None:
    """节点观测缺失、乱序或重复都会让向量切片失去固定含义，必须拒绝。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    snapshot = DPPOStateSnapshot.zeros(dimensions)
    observations = snapshot.node_observations
    if mutation == "missing":
        invalid = observations[:-1]
    elif mutation == "reordered":
        invalid = tuple(reversed(observations))
    else:
        invalid = observations[:-1] + (observations[0],)

    with pytest.raises(ValueError, match="node observations"):
        DPPOStateEncoder(dimensions).encode(
            replace(snapshot, node_observations=invalid),
            DPPOHistoryObservation.zeros(),
        )


def test_function_observations_must_match_configured_order() -> None:
    """VNF 静态特征必须与联合动作使用完全相同的函数顺序。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (10, 20))
    snapshot = DPPOStateSnapshot.zeros(dimensions)

    with pytest.raises(ValueError, match="function observations"):
        DPPOStateEncoder(dimensions).encode(
            replace(
                snapshot,
                function_observations=tuple(reversed(snapshot.function_observations)),
            ),
            DPPOHistoryObservation.zeros(),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("route_progress", np.nan),
        ("normalized_peak_requests", 1.01),
        ("global_failure_risk", -0.01),
    ],
)
def test_snapshot_rejects_non_finite_or_out_of_range_values(
    field_name: str,
    invalid_value: float,
) -> None:
    """进入策略网络之前，所有动态状态值都必须处于规定范围。"""

    dimensions = ScenarioDimensions((0, 1, 2), 3, (0,))
    snapshot = replace(
        DPPOStateSnapshot.zeros(dimensions),
        **{field_name: invalid_value},
    )

    with pytest.raises(ValueError, match="state feature"):
        DPPOStateEncoder(dimensions).encode(
            snapshot,
            DPPOHistoryObservation.zeros(),
        )


def test_history_rejects_out_of_range_ratio() -> None:
    """历史统计同样必须先归一化，避免训练中悄悄产生异常梯度。"""

    invalid_history = replace(
        DPPOHistoryObservation.zeros(),
        previous_projection_change_ratio=1.01,
    )

    with pytest.raises(ValueError, match="history feature"):
        DPPOHistoryEncoder().encode(invalid_history)
