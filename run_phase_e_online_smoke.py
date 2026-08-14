"""在 VS Code 终端运行 GPU DPPO + CPU CLARABEL 的真实双时间尺度最小闭环。"""

import argparse

import numpy as np
import torch

from src.config import load_config
from src.cost_ledger import CostLedger
from src.dppo import DPPOAgent
from src.dppo_diffusion import ConditionalDiffusionMLP, CosineNoiseSchedule
from src.dppo_training_config import build_dppo_agent_config
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
from src.phase_e_observation_adapter import PhaseEObservationAdapter
from src.phase_e_online_trainer import PhaseEOnlineTrainer
from src.queue_manager import QueueStateManager
from src.queue_state import StageFlowConfig
from src.topology import build_linear_topology


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="阶段 E 双时间尺度在线冒烟训练")
    parser.add_argument("--config", default="configs/debug.yaml")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--clip-ratio", type=float, default=0.01)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(arguments)


def _network_links(raw: dict, mec_count: int, include_cloud: bool):
    network = raw["network"]
    links: list[NetworkLinkSnapshot] = []
    link_id = 0
    edge_capacity = float(network["adjacent_bandwidth_mbps"]) * 1e6
    edge_delay = float(network["propagation_delay_per_hop_ms"]) / 1000.0
    edge_price = float(network["edge_data_cost_per_mb_hop"]) / 8e6
    for node_id in range(mec_count - 1):
        for source, destination in ((node_id, node_id + 1), (node_id + 1, node_id)):
            links.append(NetworkLinkSnapshot(
                link_id, source, destination, edge_capacity, edge_delay, edge_price
            ))
            link_id += 1
    if include_cloud:
        cloud_id = mec_count
        cloud_capacity = float(network["cloud_backhaul_bandwidth_mbps"]) * 1e6
        cloud_delay = float(network["cloud_one_way_propagation_delay_ms"]) / 1000.0
        cloud_price = float(network["cloud_data_cost_per_mb"]) / 8e6
        for mec_id in range(mec_count):
            for source, destination in ((mec_id, cloud_id), (cloud_id, mec_id)):
                links.append(NetworkLinkSnapshot(
                    link_id, source, destination,
                    cloud_capacity, cloud_delay, cloud_price,
                ))
                link_id += 1
    return tuple(links)


def main(arguments: list[str] | None = None) -> None:
    args = parse_arguments(arguments)
    if args.frames <= 0:
        raise SystemExit("--frames 必须为正整数。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用，请先激活 rail-dppo-gpu 环境。")
    raw = load_config(args.config)
    config = load_phase_a_config(raw)
    topology = build_linear_topology(raw)
    memory = {
        item["function_id"]: float(item["memory_mb"])
        for item in raw["rl_scenario"]["functions"]
    }
    function_ids = tuple(sorted(memory))
    lifecycle = InstanceLifecycleManager(config=config, function_memory_mb=memory)
    ledger = CostLedger()
    phase_a = PhaseASlotCoordinator(
        ScriptedFailureProcess(topology), lifecycle, ledger
    )
    flow = StageFlowConfig(tuple(
        float(item["output_ratio"]) for item in raw["rl_scenario"]["functions"]
    ))
    queues = QueueStateManager(
        flow,
        slot_seconds=config.fast_slot_seconds,
        # 快层 residual_tolerance 使用 Mbit，提交端必须换回 bit 后同口径审计。
        flow_absolute_tolerance_bits=(
            float(raw["fast_resource_optimization"]["residual_tolerance"])
            * 1e6
        ),
    )
    optimizer = FastResourceOptimizer(load_fast_resource_config(raw), flow)
    controller = PhaseEMainController(config, memory)
    reliability_target = float(raw["rl_scenario"]["sfc"]["reliability_target"])
    minimum_domains = int(raw["reliability"]["minimum_distinct_fault_domains"])
    fault_domain_by_node = {
        node.node_id: node.fault_domain for node in topology.compute_nodes
    }
    runtime = raw["phase_e_runtime"]
    environment = PhaseESlowFrameEnvironment(
        controller=controller,
        coordinator=PhaseBSlotCoordinator(phase_a, queues),
        optimizer=optimizer,
        lifecycle_manager=lifecycle,
        queue_manager=queues,
        cost_ledger=ledger,
        required_replica_nodes={item: minimum_domains for item in function_ids},
        fault_domain_by_node=fault_domain_by_node,
        minimum_fault_domains={item: minimum_domains for item in function_ids},
        domain_availability={
            domain_id: rates.steady_availability
            for domain_id, rates in config.domain_rates.items()
        },
        node_conditional_availability={
            node_id: rates.steady_availability
            for node_id, rates in config.node_rates.items()
        },
        maximum_vnf_unavailability={
            item: (1.0 - reliability_target) / len(function_ids)
            for item in function_ids
        },
        reference_cost=float(runtime["reference_cost_per_slow_frame"]),
    )
    mec_count = int(raw["topology"]["mec_count"])
    links = _network_links(raw, mec_count, raw["topology"].get("include_cloud") is True)
    total_slots = args.frames * config.slow_frame_slots
    observation_adapter = PhaseEObservationAdapter(
        config, controller.action_spec,
        total_episode_slots=total_slots,
        maximum_drain_slots=0,
    )
    trainer: PhaseEOnlineTrainer | None = None
    current_decision = None
    current_state = None
    previous_result = None
    update_metrics = None

    for frame_index in range(args.frames):
        start_slot = frame_index * config.slow_frame_slots

        def policy(context):
            nonlocal trainer, current_decision, current_state, update_metrics
            observation = observation_adapter.encode(
                context, previous_frame=previous_result
            )
            state = np.asarray(observation.values, dtype=np.float32)
            if trainer is None:
                dppo_config = build_dppo_agent_config(raw, clip_ratio=args.clip_ratio)
                policy_model = ConditionalDiffusionMLP(
                    observation.spec.dimension,
                    controller.action_spec.action_dim,
                    tuple(raw["dppo"]["diffusion"]["hidden_dims"]),
                )
                agent = DPPOAgent(
                    policy_model,
                    CosineNoiseSchedule(dppo_config.diffusion_steps),
                    dppo_config,
                    device=args.device,
                )
                trainer = PhaseEOnlineTrainer(
                    agent,
                    rollout_length_slow_frames=int(
                        raw["dppo"]["training"]["rollout_length_slow_frames"]
                    ),
                )
            update_metrics = trainer.update_if_ready(state)
            current_decision = trainer.sample_decision(
                state,
                seed=int(raw["dppo"]["training"]["seed"]) + frame_index,
            )
            current_state = state
            return current_decision.scores

        result = environment.run_slow_frame(
            start_slot=start_slot,
            policy=policy,
            arrivals_by_slot={
                start_slot: (
                    ArrivalBatch(
                        f"smoke-{frame_index}",
                        0,
                        float(runtime["smoke_arrival_equivalent_bits"]),
                        start_slot * config.fast_slot_seconds
                        + float(runtime["smoke_deadline_seconds"]),
                    ),
                )
            },
            network_for_slot=lambda slot: NetworkSnapshot(
                version=slot + 1,
                current_slot=slot,
                serving_mec=min(mec_count - 1, frame_index),
                channel_gain=float(runtime["channel_gain"]),
                uplink_bandwidth_hz=float(
                    raw["fast_resource_optimization"]["uplink_bandwidth_hz"]
                ),
                maximum_uplink_power_watt=float(
                    raw["fast_resource_optimization"]["maximum_uplink_power_watt"]
                ),
                links=links,
            ),
            training_mode=True,
        )
        discarded = trainer.record_result(
            current_decision,
            result,
            terminated=frame_index == args.frames - 1,
        )
        if result.code != "OK":
            raise SystemExit(
                f"frame={frame_index} internal_failure={result.code} "
                f"discarded_transitions={discarded}"
            )
        mean_fast_ms = 1000.0 * sum(
            item.fast_result.optimization.primary_solve_seconds
            + item.fast_result.optimization.secondary_solve_seconds
            for item in result.slots
        ) / len(result.slots)
        print(
            f"frame={frame_index + 1}/{args.frames} "
            f"reward={result.reward.reward:.6f} "
            f"violation={result.reward.violation_rate:.6f} "
            f"deficit={result.reward.deficit_rate:.6f} "
            f"cost={result.raw_cost:.6f} "
            f"fast_mean_ms={mean_fast_ms:.3f} "
            f"policy_device={trainer.agent.device}",
            flush=True,
        )
        if update_metrics is not None:
            print(
                f"update={trainer.completed_update_count} "
                f"policy_loss={update_metrics['policy_loss']:.6f} "
                f"value_loss={update_metrics['value_loss']:.6f} "
                f"max_kl={update_metrics['maximum_approximate_kl']:.6f}",
                flush=True,
            )
        previous_result = result

    if trainer.pending_count == trainer.rollout_length_slow_frames:
        final_metrics = trainer.update_if_ready(current_state)
        print(
            f"update={trainer.completed_update_count} "
            f"policy_loss={final_metrics['policy_loss']:.6f} "
            f"value_loss={final_metrics['value_loss']:.6f} "
            f"max_kl={final_metrics['maximum_approximate_kl']:.6f}",
            flush=True,
        )
    elif trainer.pending_count:
        print(
            f"clean_tail_frames={trainer.pending_count}（不足一个事务 rollout，未更新）",
            flush=True,
        )


if __name__ == "__main__":
    main()
