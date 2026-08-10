"""测试算法无关慢尺度执行核心的因果边界和窗口推进。"""

from src.config import load_config
from src.dppo_scenario import build_dppo_environment


def test_core_pre_generates_trace_but_exposes_only_observed_prefix() -> None:
    """未来请求保存在核心内部，预测器只能读取已经执行的请求前缀。"""

    environment = build_dppo_environment(load_config("configs/debug.yaml"))
    core = environment.execution_core

    observation = core.reset(seed=123)

    assert core.total_fast_slots > 0
    assert core.current_slot == 0
    assert core.observed_request_counts == ()
    assert observation.train_state.time_slot == core.current_slot


def test_rejected_window_advances_without_calling_fast_executor() -> None:
    """投影失败时核心仍推进外生轨迹并记录硬违约。"""

    environment = build_dppo_environment(load_config("configs/debug.yaml"))
    core = environment.execution_core
    core.reset(seed=123)
    start_slot = core.current_slot

    execution = core.advance_rejected_window(("没有可行部署",))

    assert execution.rejected is True
    assert execution.rejection_reasons == ("没有可行部署",)
    assert execution.window_start_slot == start_slot
    assert execution.metrics.sla_violations >= 1
    assert execution.reward_breakdown.violation_penalty == 1.0
    assert len(core.observed_request_counts) == execution.window_length
    assert core.current_slot > start_slot or execution.terminated


def test_projection_inputs_cover_every_configured_compute_node() -> None:
    """投影器看到的运行状态、余量和故障域必须使用同一节点集合。"""

    environment = build_dppo_environment(load_config("configs/debug.yaml"))
    core = environment.execution_core
    core.reset(seed=123)

    inputs = core.current_projection_inputs()
    expected_nodes = set(environment.dimensions.compute_node_ids)

    assert set(inputs.free_cpu) == expected_nodes
    assert set(inputs.free_memory_mb) == expected_nodes
    assert set(inputs.fault_domains) == expected_nodes
    assert inputs.operational_node_ids <= expected_nodes
