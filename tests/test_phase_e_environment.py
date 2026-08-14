"""测试阶段 E 的一个完整慢帧确实连接慢层、快层、EDF 与奖励。"""

import numpy as np
import torch

from src.config import load_config
from src.cost_ledger import CostLedger
from src.failure_process import ScriptedFailureProcess
from src.fast_resource_model import (
    NetworkLinkSnapshot,
    NetworkSnapshot,
    load_fast_resource_config,
)
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import InstanceLifecycleManager
from src.orchestration_config import load_phase_a_config
from src.orchestration_core import PhaseASlotCoordinator, PhaseBSlotCoordinator
from src.phase_e_environment import ArrivalBatch, PhaseESlowFrameEnvironment
from src.phase_e_main_controller import PhaseEMainController
from src.queue_manager import QueueStateManager
from src.queue_state import StageFlowConfig
from src.topology import build_linear_topology
from run_phase_e_online_smoke import main as run_online_smoke


def test_one_slow_frame_runs_every_fast_slot_and_returns_bounded_reward() -> None:
    raw = load_config("configs/debug.yaml")
    config = load_phase_a_config(raw)
    topology = build_linear_topology(raw)
    memory = {
        item["function_id"]: float(item["memory_mb"])
        for item in raw["rl_scenario"]["functions"]
    }
    lifecycle = InstanceLifecycleManager(config=config, function_memory_mb=memory)
    ledger = CostLedger()
    phase_a = PhaseASlotCoordinator(
        ScriptedFailureProcess(topology), lifecycle, ledger
    )
    flow = StageFlowConfig(
        tuple(float(item["output_ratio"]) for item in raw["rl_scenario"]["functions"])
    )
    queues = QueueStateManager(flow, slot_seconds=config.fast_slot_seconds)
    optimizer = FastResourceOptimizer(load_fast_resource_config(raw), flow)
    controller = PhaseEMainController(config, memory)

    links = tuple(
        NetworkLinkSnapshot(
            link_id=2 * node_id + direction,
            source_node_id=node_id if direction == 0 else node_id + 1,
            destination_node_id=node_id + 1 if direction == 0 else node_id,
            capacity_bits_per_second=10e6,
            propagation_delay_seconds=0.1,
            price_per_bit=1e-9,
        )
        for node_id in range(len(config.node_resources) - 1)
        for direction in (0, 1)
    )
    environment = PhaseESlowFrameEnvironment(
        controller=controller,
        coordinator=PhaseBSlotCoordinator(phase_a, queues),
        optimizer=optimizer,
        lifecycle_manager=lifecycle,
        queue_manager=queues,
        cost_ledger=ledger,
        required_replica_nodes={0: 1, 1: 1, 2: 1},
        reference_cost=100.0,
    )
    seen_versions = []

    result = environment.run_slow_frame(
        start_slot=0,
        policy=lambda context: (
            seen_versions.append(context.versions),
            np.zeros(controller.action_spec.action_dim),
        )[1],
        arrivals_by_slot={
            0: (ArrivalBatch("batch-0", 0, 2e5, 10.0),),
        },
        network_for_slot=lambda slot: NetworkSnapshot(
            version=slot,
            current_slot=slot,
            serving_mec=0,
            channel_gain=1e-6,
            uplink_bandwidth_hz=1e6,
            maximum_uplink_power_watt=1.0,
            links=links,
        ),
        training_mode=True,
    )

    assert result.code == "OK"
    assert result.reward is not None
    assert -1.0 <= result.reward.reward <= 0.0
    assert len(result.slots) == config.slow_frame_slots
    assert all(item.fast_result.optimization.succeeded for item in result.slots)
    assert len(seen_versions) == 1
    assert seen_versions[0].queue_version > 0  # 慢层观察已经包含本时隙新到达。
    assert result.raw_cost >= 0.0
    assert result.queue_equivalent_bits >= result.deficit_equivalent_bits >= 0.0


def test_real_environment_completes_two_transactional_ppo_rollouts(
    capsys,
) -> None:
    torch.manual_seed(3)

    run_online_smoke(["--frames", "32", "--device", "cpu"])

    output = capsys.readouterr().out
    assert "internal_failure=" not in output
    assert "update=1 " in output
    assert "update=2 " in output
