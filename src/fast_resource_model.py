"""阶段 C 快层求解器的不可变资源配置与网络快照。"""

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping


def _positive(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} 必须是正有限数。")


def _nonnegative(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} 必须是非负有限数。")


@dataclass(frozen=True)
class NodeFastResource:
    node_id: int
    maximum_cpu_cycles_per_second: float
    core_count: int
    dvfs_kappa: float
    cpu_price_per_second: float
    cloud_price_per_gcycle: float

    def __post_init__(self) -> None:
        if self.node_id < 0 or self.core_count <= 0:
            raise ValueError("节点 ID 不能为负且核心数必须为正。")
        _positive(self.maximum_cpu_cycles_per_second, "node CPU capacity")
        _positive(self.dvfs_kappa, "dvfs_kappa")
        _positive(self.cpu_price_per_second, "cpu price")
        _nonnegative(self.cloud_price_per_gcycle, "cloud price")


@dataclass(frozen=True)
class VNFComputeResource:
    function_id: int
    node_id: int
    cpu_cycles_per_physical_bit: float
    single_instance_max_cpu_cycles_per_second: float

    def __post_init__(self) -> None:
        if self.function_id < 0 or self.node_id < 0:
            raise ValueError("VNF 和节点 ID 不能为负。")
        _positive(self.cpu_cycles_per_physical_bit, "VNF cycles per bit")
        _positive(
            self.single_instance_max_cpu_cycles_per_second,
            "single instance CPU capacity",
        )


@dataclass(frozen=True)
class NetworkLinkSnapshot:
    link_id: int
    source_node_id: int
    destination_node_id: int
    capacity_bits_per_second: float
    propagation_delay_seconds: float
    price_per_bit: float

    def __post_init__(self) -> None:
        if min(self.link_id, self.source_node_id, self.destination_node_id) < 0:
            raise ValueError("链路和节点 ID 不能为负。")
        if self.source_node_id == self.destination_node_id:
            raise ValueError("物理链路两端不能相同。")
        _positive(self.capacity_bits_per_second, "link capacity")
        _positive(self.propagation_delay_seconds, "link propagation delay")
        _nonnegative(self.price_per_bit, "link price")


@dataclass(frozen=True)
class NetworkSnapshot:
    version: int
    current_slot: int
    serving_mec: int
    channel_gain: float
    uplink_bandwidth_hz: float
    maximum_uplink_power_watt: float
    links: tuple[NetworkLinkSnapshot, ...]

    def __post_init__(self) -> None:
        if min(self.version, self.current_slot, self.serving_mec) < 0:
            raise ValueError("网络版本、时隙和服务 MEC 不能为负。")
        _nonnegative(self.channel_gain, "channel gain")
        _nonnegative(self.uplink_bandwidth_hz, "uplink bandwidth")
        _nonnegative(self.maximum_uplink_power_watt, "maximum uplink power")
        if len({link.link_id for link in self.links}) != len(self.links):
            raise ValueError("link_id 必须唯一。")


@dataclass(frozen=True)
class FastResourceConfig:
    slot_seconds: float
    noise_psd_watt_per_hz: float
    energy_price_per_joule: float
    absolute_lex_tolerance: float
    relative_lex_tolerance: float
    residual_tolerance: float
    active_time_tolerance_seconds: float
    solver_name: str
    max_iterations: int
    allow_optimal_inaccurate: bool
    nodes: Mapping[int, NodeFastResource]
    vnfs: Mapping[tuple[int, int], VNFComputeResource]

    def __post_init__(self) -> None:
        for value, name in (
            (self.slot_seconds, "slot_seconds"),
            (self.noise_psd_watt_per_hz, "noise PSD"),
            (self.energy_price_per_joule, "energy price"),
            (self.absolute_lex_tolerance, "absolute lex tolerance"),
            (self.relative_lex_tolerance, "relative lex tolerance"),
            (self.residual_tolerance, "residual tolerance"),
            (self.active_time_tolerance_seconds, "active time tolerance"),
        ):
            _positive(value, name)
        if self.solver_name != "CLARABEL":
            raise ValueError("阶段 C 求解器必须为 CLARABEL。")
        if isinstance(self.max_iterations, bool) or self.max_iterations <= 0:
            raise ValueError("max_iterations 必须是正整数。")
        nodes = dict(self.nodes)
        vnfs = dict(self.vnfs)
        if set(nodes) != {item.node_id for item in nodes.values()}:
            raise ValueError("nodes 映射键必须等于资源 node_id。")
        if set(vnfs) != {
            (item.function_id, item.node_id) for item in vnfs.values()
        }:
            raise ValueError("vnfs 映射键必须等于 VNF—节点 ID。")
        if any(node_id not in nodes for _, node_id in vnfs):
            raise ValueError("VNF 资源引用了未知节点。")
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        object.__setattr__(self, "vnfs", MappingProxyType(vnfs))


def load_fast_resource_config(config: dict[str, Any]) -> FastResourceConfig:
    """从场景规模和阶段 A 部署组合生成快层资源配置。"""

    section = config["fast_resource_optimization"]
    resources = config["node_resources"]
    topology = config["topology"]
    mec_count = int(topology["mec_count"])
    include_cloud = topology.get("include_cloud") is True
    nodes: dict[int, NodeFastResource] = {}
    for node_id in range(mec_count + int(include_cloud)):
        is_cloud = node_id == mec_count
        nodes[node_id] = NodeFastResource(
            node_id=node_id,
            maximum_cpu_cycles_per_second=float(
                resources[
                    "cloud_cpu_capacity_cycles_per_second"
                    if is_cloud else "mec_cpu_capacity_cycles_per_second"
                ]
            ),
            core_count=int(
                resources["cloud_core_count" if is_cloud else "mec_core_count"]
            ),
            dvfs_kappa=float(
                section["cloud_dvfs_kappa" if is_cloud else "mec_dvfs_kappa"]
            ),
            cpu_price_per_second=float(
                section[
                    "cloud_cpu_price_per_second"
                    if is_cloud else "mec_cpu_price_per_second"
                ]
            ),
            cloud_price_per_gcycle=(
                float(section["cloud_price_per_gcycle"]) if is_cloud else 0.0
            ),
        )
    functions = {
        int(item["function_id"]): item
        for item in config["rl_scenario"]["functions"]
    }
    pairs: dict[tuple[int, int], VNFComputeResource] = {}
    for deployment in config["instance_lifecycle"]["allowed_deployments"]:
        function_id = int(deployment["function_id"])
        for node_id in deployment["node_ids"]:
            key = (function_id, int(node_id))
            pairs[key] = VNFComputeResource(
                function_id,
                int(node_id),
                float(functions[function_id]["cpu_cycles_per_input_bit"]),
                float(deployment["single_instance_max_cpu_cycles_per_second"]),
            )
    return FastResourceConfig(
        slot_seconds=float(config["simulation"]["fast_slot_seconds"]),
        noise_psd_watt_per_hz=float(section["noise_psd_watt_per_hz"]),
        energy_price_per_joule=float(section["energy_price_per_joule"]),
        absolute_lex_tolerance=float(section["absolute_lex_tolerance"]),
        relative_lex_tolerance=float(section["relative_lex_tolerance"]),
        residual_tolerance=float(section["residual_tolerance"]),
        active_time_tolerance_seconds=float(
            section["active_time_tolerance_seconds"]
        ),
        solver_name=str(section["solver"]),
        max_iterations=int(section["max_iterations"]),
        allow_optimal_inaccurate=bool(section["allow_optimal_inaccurate"]),
        nodes=nodes,
        vnfs=pairs,
    )
