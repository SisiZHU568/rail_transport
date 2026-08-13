"""阶段 A 使用的只读编排配置对象和严格解析入口。"""

from dataclasses import dataclass
import math
from typing import Any


def _finite_number(value: object, field: str, *, minimum: float = 0.0) -> float:
    """读取有限数，并在错误中保留完整配置路径。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是数值。")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} 必须是不小于 {minimum} 的有限数。")
    return result


def _positive_integer(value: object, field: str) -> int:
    """读取严格为正的整数，拒绝布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数。")
    return value


@dataclass(frozen=True)
class CTMCRates:
    """一个组件的连续时间失效率和恢复率，单位均为 1/秒。"""

    failure_rate_per_second: float
    recovery_rate_per_second: float

    def __post_init__(self) -> None:
        values = (self.failure_rate_per_second, self.recovery_rate_per_second)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("CTMC 失效率和恢复率必须是非负有限数。")
        if sum(values) <= 0.0:
            raise ValueError("CTMC 失效率与恢复率不能同时为零。")

    @property
    def steady_availability(self) -> float:
        """返回连续时间两状态链的稳态可用度。"""

        return self.recovery_rate_per_second / (
            self.failure_rate_per_second + self.recovery_rate_per_second
        )


@dataclass(frozen=True)
class LifecycleConfig:
    """实例生命周期的结构版本、价格版本与保留时间范围。"""

    schema_version: str
    price_version: str
    minimum_retention_seconds: float
    maximum_retention_seconds: float
    retention_step_seconds: float

    def __post_init__(self) -> None:
        if not self.schema_version or not self.price_version:
            raise ValueError("instance_lifecycle 的版本必须是非空字符串。")
        values = (
            self.minimum_retention_seconds,
            self.maximum_retention_seconds,
            self.retention_step_seconds,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("instance_lifecycle 保留时间必须是有限数。")
        if self.minimum_retention_seconds < 0.0:
            raise ValueError("minimum_retention_seconds 不能小于零。")
        if self.maximum_retention_seconds < self.minimum_retention_seconds:
            raise ValueError("maximum_retention_seconds 不能小于最小值。")
        if self.retention_step_seconds <= 0.0:
            raise ValueError("retention_step_seconds 必须大于零。")


@dataclass(frozen=True)
class DeploymentPairConfig:
    """一个允许部署的 VNF—节点组合及其物理/价格边界。"""

    function_id: int
    node_id: int
    minimum_instances: int
    maximum_instances: int
    cold_start_seconds: float
    single_instance_max_cpu_cycles_per_second: float
    deployment_cost_per_instance: float
    cold_start_cost_per_instance: float
    running_cost_per_instance_second: float

    @property
    def allowed_instance_counts(self) -> tuple[int, ...]:
        """零表示不部署，正数区间表示部署后的合法容器数。"""

        return (0, *range(self.minimum_instances, self.maximum_instances + 1))


@dataclass(frozen=True)
class NodeResourceConfig:
    """一个部署节点的内存、总 CPU 周期率与核心数。"""

    node_id: int
    memory_capacity_mb: float
    cpu_capacity_cycles_per_second: float
    core_count: int


@dataclass(frozen=True)
class PhaseAConfig:
    """阶段 A 跨模块共享的不可变配置。"""

    fast_slot_seconds: float
    failure_base_seed: int
    domain_rates: dict[int, CTMCRates]
    node_rates: dict[int, CTMCRates]
    node_resources: dict[int, NodeResourceConfig]
    deployment_pairs: dict[tuple[int, int], DeploymentPairConfig]
    lifecycle: LifecycleConfig

    @property
    def retention_slot_options(self) -> tuple[int, ...]:
        """按上取整生成规范保留档，并强制包含最大值。"""

        lifecycle = self.lifecycle
        seconds = {lifecycle.maximum_retention_seconds}
        index = 0
        while (
            lifecycle.minimum_retention_seconds
            + index * lifecycle.retention_step_seconds
            <= lifecycle.maximum_retention_seconds
        ):
            seconds.add(
                lifecycle.minimum_retention_seconds
                + index * lifecycle.retention_step_seconds
            )
            index += 1
        return tuple(
            sorted(
                {
                    math.ceil(value / self.fast_slot_seconds)
                    for value in seconds
                }
            )
        )

    def allowed_instance_counts(
        self,
        function_id: int,
        node_id: int,
    ) -> tuple[int, ...]:
        """返回指定 VNF—节点组合的合法目标实例数。"""

        return self.deployment_pairs[(function_id, node_id)].allowed_instance_counts


def _rate_map(items: object, field: str) -> dict[int, CTMCRates]:
    """把显式 ID 列表解析成唯一 CTMC 率字典。"""

    if not isinstance(items, list) or not items:
        raise ValueError(f"runtime_failure.{field} 必须是非空列表。")
    result: dict[int, CTMCRates] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"runtime_failure.{field} 条目必须是字典。")
        entity_id = item.get("id")
        if (
            isinstance(entity_id, bool)
            or not isinstance(entity_id, int)
            or entity_id < 0
            or entity_id in result
        ):
            raise ValueError(f"runtime_failure.{field} 的 id 必须唯一且非负。")
        try:
            failure_rate = _finite_number(
                item.get("failure_rate_per_second"),
                f"runtime_failure.{field}.failure_rate_per_second",
            )
            recovery_rate = _finite_number(
                item.get("recovery_rate_per_second"),
                f"runtime_failure.{field}.recovery_rate_per_second",
            )
            result[entity_id] = CTMCRates(failure_rate, recovery_rate)
        except ValueError as error:
            raise ValueError(f"runtime_failure.{field} 包含无效 CTMC 率。") from error
    return result


def load_ctmc_rate_maps(
    config: dict[str, Any],
) -> tuple[dict[int, CTMCRates], dict[int, CTMCRates]]:
    """读取域/节点连续时间率，供拓扑与可靠性共享同一数值来源。"""

    runtime = config["runtime_failure"]
    return (
        _rate_map(runtime.get("domain_rates"), "domain_rates"),
        _rate_map(runtime.get("node_rates"), "node_rates"),
    )


def _node_resources(config: dict[str, Any]) -> dict[int, NodeResourceConfig]:
    """按规范节点 ID 展开同构 MEC 资源和中心云资源。"""

    topology = config["topology"]
    resources = config["node_resources"]
    mec_count = _positive_integer(topology.get("mec_count"), "topology.mec_count")
    result = {
        node_id: NodeResourceConfig(
            node_id=node_id,
            memory_capacity_mb=_finite_number(
                resources.get("mec_memory_mb"),
                "node_resources.mec_memory_mb",
                minimum=1e-300,
            ),
            cpu_capacity_cycles_per_second=_finite_number(
                resources.get("mec_cpu_capacity_cycles_per_second"),
                "node_resources.mec_cpu_capacity_cycles_per_second",
                minimum=1e-300,
            ),
            core_count=_positive_integer(
                resources.get("mec_core_count"),
                "node_resources.mec_core_count",
            ),
        )
        for node_id in range(mec_count)
    }
    if topology.get("include_cloud") is True:
        result[mec_count] = NodeResourceConfig(
            node_id=mec_count,
            memory_capacity_mb=_finite_number(
                resources.get("cloud_memory_mb"),
                "node_resources.cloud_memory_mb",
                minimum=1e-300,
            ),
            cpu_capacity_cycles_per_second=_finite_number(
                resources.get("cloud_cpu_capacity_cycles_per_second"),
                "node_resources.cloud_cpu_capacity_cycles_per_second",
                minimum=1e-300,
            ),
            core_count=_positive_integer(
                resources.get("cloud_core_count"),
                "node_resources.cloud_core_count",
            ),
        )
    return result


def _deployment_pairs(
    config: dict[str, Any],
    node_resources: dict[int, NodeResourceConfig],
) -> dict[tuple[int, int], DeploymentPairConfig]:
    """展开允许部署列表，并验证所有函数、节点和单位。"""

    lifecycle = config["instance_lifecycle"]
    items = lifecycle.get("allowed_deployments")
    if not isinstance(items, list) or not items:
        raise ValueError("instance_lifecycle.allowed_deployments 必须是非空列表。")
    functions = {
        item["function_id"]: item
        for item in config["rl_scenario"]["functions"]
    }
    result: dict[tuple[int, int], DeploymentPairConfig] = {}
    for index, item in enumerate(items):
        path = f"instance_lifecycle.allowed_deployments[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{path} 必须是字典。")
        function_id = item.get("function_id")
        node_ids = item.get("node_ids")
        if function_id not in functions:
            raise ValueError(f"{path}.function_id 引用了未知 VNF。")
        if (
            not isinstance(node_ids, list)
            or not node_ids
            or any(node_id not in node_resources for node_id in node_ids)
        ):
            raise ValueError(f"{path}.node_ids 引用了未知 node。")
        minimum = _positive_integer(item.get("minimum_instances"), f"{path}.minimum_instances")
        maximum = _positive_integer(item.get("maximum_instances"), f"{path}.maximum_instances")
        if minimum > maximum:
            raise ValueError(f"{path} 的实例数下限不能大于上限。")
        cold_start = _finite_number(item.get("cold_start_seconds"), f"{path}.cold_start_seconds")
        expected = float(functions[function_id]["cold_start_time_ms"]) / 1000.0
        if not math.isclose(cold_start, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{path}.cold_start_seconds 与 VNF 毫秒配置不一致。")
        for node_id in node_ids:
            key = (function_id, node_id)
            if key in result:
                raise ValueError(f"{path} 重复声明 VNF—节点组合 {key}。")
            result[key] = DeploymentPairConfig(
                function_id=function_id,
                node_id=node_id,
                minimum_instances=minimum,
                maximum_instances=maximum,
                cold_start_seconds=cold_start,
                single_instance_max_cpu_cycles_per_second=_finite_number(
                    item.get("single_instance_max_cpu_cycles_per_second"),
                    f"{path}.single_instance_max_cpu_cycles_per_second",
                    minimum=1e-300,
                ),
                deployment_cost_per_instance=_finite_number(
                    item.get("deployment_cost_per_instance"),
                    f"{path}.deployment_cost_per_instance",
                ),
                cold_start_cost_per_instance=_finite_number(
                    item.get("cold_start_cost_per_instance"),
                    f"{path}.cold_start_cost_per_instance",
                ),
                running_cost_per_instance_second=_finite_number(
                    item.get("running_cost_per_instance_second"),
                    f"{path}.running_cost_per_instance_second",
                ),
            )
    missing_functions = set(functions) - {key[0] for key in result}
    if missing_functions:
        raise ValueError(
            "instance_lifecycle.allowed_deployments 缺少 VNF："
            f"{sorted(missing_functions)}。"
        )
    return result


def load_phase_a_config(config: dict[str, Any]) -> PhaseAConfig:
    """从总配置构造阶段 A 的唯一只读模型。"""

    simulation = config["simulation"]
    runtime = config["runtime_failure"]
    lifecycle = config["instance_lifecycle"]
    slot_seconds = _finite_number(
        simulation.get("fast_slot_seconds"),
        "simulation.fast_slot_seconds",
        minimum=1e-300,
    )
    base_seed = runtime.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("runtime_failure.base_seed 必须是非负整数。")
    resources = _node_resources(config)
    domain_rates, node_rates = load_ctmc_rate_maps(config)
    expected_nodes = set(resources)
    if set(node_rates) != expected_nodes:
        raise ValueError("runtime_failure.node_rates 必须覆盖全部且仅覆盖部署节点。")
    mec_domains = set(config["topology"]["mec_fault_domain_ids"])
    if config["topology"].get("include_cloud") is True:
        mec_domains.add(config["node_resources"]["cloud_fault_domain_id"])
    if set(domain_rates) != mec_domains:
        raise ValueError("runtime_failure.domain_rates 必须覆盖全部且仅覆盖故障域。")
    result = PhaseAConfig(
        fast_slot_seconds=slot_seconds,
        failure_base_seed=base_seed,
        domain_rates=domain_rates,
        node_rates=node_rates,
        node_resources=resources,
        deployment_pairs=_deployment_pairs(config, resources),
        lifecycle=LifecycleConfig(
            schema_version=str(lifecycle.get("schema_version", "")),
            price_version=str(lifecycle.get("price_version", "")),
            minimum_retention_seconds=_finite_number(
                lifecycle.get("minimum_retention_seconds"),
                "instance_lifecycle.minimum_retention_seconds",
            ),
            maximum_retention_seconds=_finite_number(
                lifecycle.get("maximum_retention_seconds"),
                "instance_lifecycle.maximum_retention_seconds",
            ),
            retention_step_seconds=_finite_number(
                lifecycle.get("retention_step_seconds"),
                "instance_lifecycle.retention_step_seconds",
                minimum=1e-300,
            ),
        ),
    )
    return result
