"""测试 DDQN 78 维状态模式、特征顺序和输入校验。"""

from dataclasses import replace

import numpy as np
import pytest

from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.rl_agent_action_space import (
    decode_ddqn_action,
)
from src.rl_state_encoder import (
    NodeStateObservation,
    RLStateEncoder,
    RLStateSnapshot,
)


def build_functions() -> list[ServerlessFunction]:
    """创建与当前实验一致的三函数 SFC。"""

    return [
        ServerlessFunction(
            function_id=0,
            name="数据清洗",
            memory_mb=256.0,
            cpu_cycles_per_request=20.0,
            image_size_mb=80.0,
            warm_exec_time_ms=15.0,
            cold_start_time_ms=300.0,
            output_ratio=0.7,
        ),
        ServerlessFunction(
            function_id=1,
            name="特征提取",
            memory_mb=512.0,
            cpu_cycles_per_request=40.0,
            image_size_mb=150.0,
            warm_exec_time_ms=30.0,
            cold_start_time_ms=500.0,
            output_ratio=0.4,
        ),
        ServerlessFunction(
            function_id=2,
            name="异常检测",
            memory_mb=768.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=300.0,
            warm_exec_time_ms=50.0,
            cold_start_time_ms=800.0,
            output_ratio=0.1,
        ),
    ]


def build_encoder() -> RLStateEncoder:
    """使用清晰的上限创建当前五 MEC、三函数编码器。"""

    return RLStateEncoder(
        trackside_node_ids=(0, 1, 2, 3, 4),
        cloud_node_id=5,
        functions=build_functions(),
        sfc=SFCType(
            sfc_id=0,
            name="高可靠列车设备故障诊断",
            function_ids=[0, 1, 2],
            deadline_ms=1500.0,
            reliability_target=0.99,
            priority=ServicePriority.CRITICAL,
        ),
        input_size_mb_per_request=2.0,
        maximum_cpu_cycles_per_request=100.0,
        maximum_function_memory_mb=1000.0,
        maximum_execution_time_ms=100.0,
        maximum_cold_start_time_ms=1000.0,
        maximum_deadline_ms=2000.0,
        maximum_input_size_mb=10.0,
        maximum_replica_count=3,
    )


def build_snapshot(
    *,
    current_action_id: int | None = 9,
    has_history: bool = True,
) -> RLStateSnapshot:
    """创建各分组数值容易核对的状态快照。"""

    observations = tuple(
        NodeStateObservation(
            node_id=node_id,
            free_cpu_ratio=0.90 - 0.05 * node_id,
            free_memory_ratio=0.80 - 0.05 * node_id,
            base_availability=0.99,
            predicted_failure_probability=0.01,
            operational=node_id != 4,
            normalized_delay_from_serving=0.10 * node_id,
        )
        for node_id in range(6)
    )

    return RLStateSnapshot(
        serving_mec=1,
        next_mec=2,
        route_progress=0.25,
        normalized_remaining_dwell=0.50,
        normalized_mean_requests=0.40,
        normalized_peak_requests=0.60,
        normalized_load_trend=-0.50,
        global_failure_risk=0.10,
        node_observations=observations,
        current_action=(
            None
            if current_action_id is None
            else decode_ddqn_action(
                current_action_id
            )
        ),
        hot_replica_ratio=0.50,
        cloud_replica_ratio=0.25,
        has_history=has_history,
        previous_success_rate=0.90,
        previous_sla_violation_rate=0.10,
        previous_normalized_total_cost=0.20,
        previous_repair_failure_rate=0.05,
    )


def test_current_schema_has_seventy_eight_features() -> None:
    """当前 N=5、M=3 的状态必须稳定为 78 维。"""

    encoder = build_encoder()
    state = encoder.encode(build_snapshot())

    assert encoder.state_dim == 78
    assert len(encoder.feature_names) == 78
    assert state.shape == (78,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert ((0.0 <= state) & (state <= 1.0)).all()


def test_feature_groups_have_stable_names_and_values() -> None:
    """检查四组特征的边界，防止以后修改时顺序漂移。"""

    encoder = build_encoder()
    state = encoder.encode(build_snapshot())

    assert encoder.feature_names[:5] == tuple(
        f"current_mec_{node_id}"
        for node_id in range(5)
    )
    assert encoder.feature_names[5:10] == tuple(
        f"next_mec_{node_id}"
        for node_id in range(5)
    )
    assert encoder.feature_names[16] == (
        "node_0_free_cpu_ratio"
    )
    assert encoder.feature_names[52] == (
        "function_0_normalized_cpu"
    )
    assert encoder.feature_names[64:67] == (
        "sfc_reliability_target",
        "sfc_normalized_deadline",
        "sfc_normalized_input_size",
    )
    assert encoder.feature_names[67] == (
        "current_normalized_replica_count"
    )

    np.testing.assert_array_equal(
        state[:5],
        np.array([0, 1, 0, 0, 0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        state[5:10],
        np.array([0, 0, 1, 0, 0], dtype=np.float32),
    )
    assert state[10] == pytest.approx(0.25)
    assert state[14] == pytest.approx(0.25)
    assert state[52] == pytest.approx(0.20)
    assert state[64] == pytest.approx(0.99)
    assert state[65] == pytest.approx(0.75)
    assert state[66] == pytest.approx(0.20)


def test_node_ids_are_one_hot_not_scalar_ordinals() -> None:
    """节点编号只能作为类别 one-hot，不能被误当成连续数值。"""

    encoder = build_encoder()
    state = encoder.encode(build_snapshot())

    assert state[:5].sum() == 1.0
    assert state[5:10].sum() == 1.0


def test_history_encodes_structured_action_components() -> None:
    """历史动作保存三个组成部分，而不是额外的 12 维 one-hot。"""

    state = build_encoder().encode(
        build_snapshot(current_action_id=9)
    )

    expected_history = np.array(
        [
            1.0,              # 3 个副本 / 最大 3 个副本
            0.0, 1.0, 0.0,   # PRIMARY_WARM one-hot
            1.0,              # CLOUD_ALLOWED
            0.50,
            0.25,
            0.90,
            0.10,
            0.20,
            0.05,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        state[67:],
        expected_history,
    )


def test_reset_history_is_all_zero() -> None:
    """Episode 首次 reset 不得伪造默认动作或历史指标。"""

    snapshot = build_snapshot(
        current_action_id=None,
        has_history=False,
    )
    state = build_encoder().encode(snapshot)

    assert np.all(state[67:] == 0.0)


@pytest.mark.parametrize(
    "observations",
    [
        build_snapshot().node_observations[:-1],
        tuple(reversed(build_snapshot().node_observations)),
        (
            build_snapshot().node_observations[:-1]
            + (build_snapshot().node_observations[0],)
        ),
    ],
)
def test_missing_reordered_or_duplicate_nodes_are_rejected(
    observations: tuple[NodeStateObservation, ...],
) -> None:
    """节点特征必须严格按五个 MEC 加中心云的固定顺序提供。"""

    snapshot = replace(
        build_snapshot(),
        node_observations=observations,
    )

    with pytest.raises(ValueError, match="节点观测"):
        build_encoder().encode(snapshot)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("route_progress", np.nan),
        ("normalized_peak_requests", 1.01),
        ("global_failure_risk", -0.01),
    ],
)
def test_non_finite_or_out_of_range_snapshot_is_rejected(
    field_name: str,
    invalid_value: float,
) -> None:
    """进入神经网络前必须拒绝非有限值和超范围值。"""

    snapshot = replace(
        build_snapshot(),
        **{field_name: invalid_value},
    )

    with pytest.raises(ValueError, match="状态特征"):
        build_encoder().encode(snapshot)


def test_history_requires_a_structured_action() -> None:
    """已有历史窗口时不能缺少对应的结构化慢层动作。"""

    snapshot = replace(
        build_snapshot(),
        current_action=None,
        has_history=True,
    )

    with pytest.raises(ValueError, match="历史动作"):
        build_encoder().encode(snapshot)
