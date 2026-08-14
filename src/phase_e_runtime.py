"""构建可供 smoke、正式训练和验证复用的 Phase E 真实运行时。"""

from dataclasses import dataclass

from src.cost_ledger import CostLedger
from src.failure_process import ScriptedFailureProcess
from src.fast_resource_model import (
    NetworkLinkSnapshot,
    NetworkSnapshot,
    load_fast_resource_config,
)
from src.fast_resource_optimizer import FastResourceOptimizer
from src.instance_lifecycle import InstanceLifecycleManager
from src.orchestration_config import PhaseAConfig, load_phase_a_config
from src.orchestration_core import PhaseASlotCoordinator, PhaseBSlotCoordinator
from src.phase_e_environment import PhaseESlowFrameEnvironment
from src.phase_e_main_controller import PhaseEMainController
from src.phase_e_observation_adapter import PhaseEObservationAdapter
from src.queue_manager import QueueStateManager
from src.queue_state import StageFlowConfig
from src.topology import build_linear_topology


@dataclass(frozen=True)
class PhaseERuntime:
    raw_config: dict
    config: PhaseAConfig
    controller: PhaseEMainController
    environment: PhaseESlowFrameEnvironment
    observation_adapter: PhaseEObservationAdapter
    network_links: tuple[NetworkLinkSnapshot, ...]
    mec_count: int
    training_seed: int

    def network_for_slot(self, slot: int, *, frame_index: int) -> NetworkSnapshot:
        if slot < 0 or frame_index < 0:
            raise ValueError("slot 和 frame_index 必须为非负整数。")
        settings = self.raw_config["phase_e_runtime"]
        resources = self.raw_config["fast_resource_optimization"]
        return NetworkSnapshot(
            version=slot + 1,
            current_slot=slot,
            serving_mec=min(self.mec_count - 1, frame_index),
            channel_gain=float(settings["channel_gain"]),
            uplink_bandwidth_hz=float(resources["uplink_bandwidth_hz"]),
            maximum_uplink_power_watt=float(resources["maximum_uplink_power_watt"]),
            links=self.network_links,
        )


def _network_links(
    raw: dict,
    mec_count: int,
    include_cloud: bool,
) -> tuple[NetworkLinkSnapshot, ...]:
    network = raw["network"]
    links: list[NetworkLinkSnapshot] = []
    link_id = 0
    edge_capacity = float(network["adjacent_bandwidth_mbps"]) * 1e6
    edge_delay = float(network["propagation_delay_per_hop_ms"]) / 1000.0
    edge_price = float(network["edge_data_cost_per_mb_hop"]) / 8e6
    for node_id in range(mec_count - 1):
        for source, destination in (
            (node_id, node_id + 1),
            (node_id + 1, node_id),
        ):
            links.append(
                NetworkLinkSnapshot(
                    link_id,
                    source,
                    destination,
                    edge_capacity,
                    edge_delay,
                    edge_price,
                )
            )
            link_id += 1
    if include_cloud:
        cloud_id = mec_count
        cloud_capacity = float(network["cloud_backhaul_bandwidth_mbps"]) * 1e6
        cloud_delay = float(network["cloud_one_way_propagation_delay_ms"]) / 1000.0
        cloud_price = float(network["cloud_data_cost_per_mb"]) / 8e6
        for mec_id in range(mec_count):
            for source, destination in ((mec_id, cloud_id), (cloud_id, mec_id)):
                links.append(
                    NetworkLinkSnapshot(
                        link_id,
                        source,
                        destination,
                        cloud_capacity,
                        cloud_delay,
                        cloud_price,
                    )
                )
                link_id += 1
    return tuple(links)


def build_phase_e_runtime(raw: dict, *, total_slow_frames: int) -> PhaseERuntime:
    if total_slow_frames <= 0:
        raise ValueError("total_slow_frames 必须为正整数。")
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
        ScriptedFailureProcess(topology),
        lifecycle,
        ledger,
    )
    flow = StageFlowConfig(
        tuple(
            float(item["output_ratio"])
            for item in raw["rl_scenario"]["functions"]
        )
    )
    queues = QueueStateManager(
        flow,
        slot_seconds=config.fast_slot_seconds,
        flow_absolute_tolerance_bits=(
            float(raw["fast_resource_optimization"]["residual_tolerance"])
            * 1e6
        ),
    )
    controller = PhaseEMainController(config, memory)
    reliability_target = float(raw["rl_scenario"]["sfc"]["reliability_target"])
    minimum_domains = int(raw["reliability"]["minimum_distinct_fault_domains"])
    fault_domain_by_node = {
        node.node_id: node.fault_domain for node in topology.compute_nodes
    }
    environment = PhaseESlowFrameEnvironment(
        controller=controller,
        coordinator=PhaseBSlotCoordinator(phase_a, queues),
        optimizer=FastResourceOptimizer(load_fast_resource_config(raw), flow),
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
        reference_cost=float(
            raw["phase_e_runtime"]["reference_cost_per_slow_frame"]
        ),
    )
    mec_count = int(raw["topology"]["mec_count"])
    links = _network_links(
        raw,
        mec_count,
        raw["topology"].get("include_cloud") is True,
    )
    observation_adapter = PhaseEObservationAdapter(
        config,
        controller.action_spec,
        total_episode_slots=total_slow_frames * config.slow_frame_slots,
        maximum_drain_slots=0,
    )
    return PhaseERuntime(
        raw,
        config,
        controller,
        environment,
        observation_adapter,
        links,
        mec_count,
        int(raw["dppo"]["training"]["seed"]),
    )
