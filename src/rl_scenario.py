"""从配置集中创建学习算法使用的轨道边缘场景。"""

from typing import Any

from src.constraint_audit import SlotConstraintAuditor
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_risk_prediction import build_windowed_failure_risk_provider
from src.fast_optimizer import FastFeasibilityOptimizer
from src.fast_slot_executor import (
    FastSlotExecutor,
    RuntimeCostRates,
)
from src.mobility import TrainMobilityModel
from src.network import build_hybrid_rail_network
from src.redundancy_placement import build_reliability_aware_replica_planner
from src.reliability import build_fault_domain_reliability_model
from src.risk_aware_failure_process import build_windowed_markov_failure_process
from src.rl_state_encoder import RLStateEncoder
from src.runtime_reliability import SingleReplicaPlanner
from src.slow_timescale_rl_env import SlowTimescaleRLEnvironment
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload
from src.workload_prediction import HistoricalWorkloadPredictor


def _scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回学习场景配置，并在缺失时给出容易定位的错误。"""

    scenario = config.get("rl_scenario")
    if not isinstance(scenario, dict):
        raise ValueError("配置项 rl_scenario 必须是字典。")
    return scenario


def _require_nonnegative_integer(value: Any, name: str) -> int:
    """拒绝布尔值、小数和字符串，避免编号在转换时被静默改变。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


def build_rl_functions(config: dict[str, Any]) -> list[ServerlessFunction]:
    """按照 YAML 列表顺序构造 VNF，不在 Python 中固定 VNF 数量。"""

    records = _scenario_config(config).get("functions")
    if not isinstance(records, list) or not records:
        raise ValueError("rl_scenario.functions 必须是非空列表。")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("每个 VNF 配置必须是字典。")

    function_ids = tuple(
        _require_nonnegative_integer(record.get("function_id"), "VNF ID")
        for record in records
    )
    if len(set(function_ids)) != len(function_ids):
        raise ValueError("VNF ID 必须唯一。")

    # 保持 records 的原始顺序；这个顺序会用于状态和动作向量切片。
    return [
        ServerlessFunction(
            function_id=function_id,
            name=str(record["name"]),
            memory_mb=float(record["memory_mb"]),
            cpu_cycles_per_request=float(record["cpu_cycles_per_request"]),
            image_size_mb=float(record["image_size_mb"]),
            warm_exec_time_ms=float(record["warm_exec_time_ms"]),
            cold_start_time_ms=float(record["cold_start_time_ms"]),
            output_ratio=float(record["output_ratio"]),
        )
        for function_id, record in zip(function_ids, records, strict=True)
    ]


def build_rl_sfc(config: dict[str, Any]) -> SFCType:
    """从配置构造 SFC，并确认链中没有未知、重复或遗漏的 VNF。"""

    scenario = _scenario_config(config)
    sfc_config = scenario.get("sfc")
    if not isinstance(sfc_config, dict):
        raise ValueError("rl_scenario.sfc 必须是字典。")

    configured_ids = tuple(
        function.function_id for function in build_rl_functions(config)
    )
    raw_chain = sfc_config.get("function_ids")
    if not isinstance(raw_chain, list) or not raw_chain:
        raise ValueError("SFC function_ids 必须是非空列表。")
    chain = tuple(
        _require_nonnegative_integer(function_id, "SFC VNF ID")
        for function_id in raw_chain
    )
    if chain != configured_ids:
        raise ValueError(
            "SFC VNF 顺序必须与 rl_scenario.functions 的配置顺序完全一致。"
        )

    try:
        priority = ServicePriority(str(sfc_config["priority"]).lower())
    except ValueError as error:
        allowed = ", ".join(priority.value for priority in ServicePriority)
        raise ValueError(f"SFC priority 必须是以下值之一：{allowed}。") from error

    return SFCType(
        sfc_id=_require_nonnegative_integer(sfc_config.get("sfc_id"), "SFC ID"),
        name=str(sfc_config["name"]),
        function_ids=list(chain),
        deadline_ms=float(sfc_config["deadline_ms"]),
        reliability_target=float(sfc_config["reliability_target"]),
        priority=priority,
    )


def _rl_subconfig(
    config: dict[str, Any],
    name: str,
    defaults: dict[str, float],
) -> dict[str, float]:
    """读取第 14 步正式落盘前也能工作的 RL 子配置。"""

    supplied = config["rl_environment"].get(name, {})
    return {
        key: float(supplied.get(key, default))
        for key, default in defaults.items()
    }


def build_rl_environment(config: dict[str, Any]) -> SlowTimescaleRLEnvironment:
    """根据配置创建当前过渡期学习环境和共享快层执行闭环。"""

    topology = build_linear_topology(config)
    if topology.cloud_node is None:
        raise ValueError("当前平铺状态模式要求 topology.include_cloud=true。")
    # 强化学习状态同时观察轨旁边缘节点和中心云，因此网络模型也必须包含
    # 中心云链路；仅构建线性 MEC 网络会导致查询中心云时延时找不到节点。
    network = build_hybrid_rail_network(config=config, topology=topology)
    functions = build_rl_functions(config)
    sfc = build_rl_sfc(config)
    slot_seconds = float(config["simulation"]["fast_slot_seconds"])
    input_size_mb_per_request = float(
        config["rl_scenario"]["sfc"]["input_size_mb_per_request"]
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=float(config["train"]["initial_position_m"]),
        speed_mps=float(config["train"]["speed_mps"]),
        slot_seconds=slot_seconds,
    )
    workload = DeterministicWorkload(
        request_trace=config["integrated_simulation"]["request_trace"],
        repeat=bool(
            config["integrated_simulation"]["repeat_request_trace"]
        ),
    )

    prediction_config = _rl_subconfig(
        config,
        "workload_prediction",
        {
            "lookback_slots": 10.0,
            "baseline_request_rate": 1.0,
        },
    )
    normalization_config = _rl_subconfig(
        config,
        "state_normalization",
        {
            "maximum_request_rate": 5.0,
            "maximum_network_delay_ms": 200.0,
            "maximum_input_size_mb": 10.0,
        },
    )
    cost_config = _rl_subconfig(
        config,
        "cost_rates",
        {
            "edge_cpu_cost_per_unit": 0.01,
            "edge_memory_cost_per_mb_second": 0.001,
            "cloud_cpu_cost_per_unit": 0.05,
            "cloud_memory_cost_per_mb_second": 0.005,
            "cold_start_cost_per_ms": 0.10,
            "maximum_window_cost": 10000.0,
        },
    )

    reliability_model = build_fault_domain_reliability_model(
        config=config,
        topology=topology,
    )
    constraint_auditor = SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )
    cost_rates = RuntimeCostRates(
        edge_cpu_cost_per_unit=cost_config["edge_cpu_cost_per_unit"],
        edge_memory_cost_per_mb_second=(
            cost_config["edge_memory_cost_per_mb_second"]
        ),
        cloud_cpu_cost_per_unit=cost_config["cloud_cpu_cost_per_unit"],
        cloud_memory_cost_per_mb_second=(
            cost_config["cloud_memory_cost_per_mb_second"]
        ),
        cold_start_cost_per_ms=cost_config["cold_start_cost_per_ms"],
    )
    fast_optimizer = FastFeasibilityOptimizer(
        functions=functions,
        sfc=sfc,
        topology=topology,
        auditor=constraint_auditor,
        return_result_to_source=True,
        network=network,
        edge_cpu_cost_per_unit=cost_rates.edge_cpu_cost_per_unit,
        edge_memory_cost_per_mb_second=(
            cost_rates.edge_memory_cost_per_mb_second
        ),
        cloud_cpu_cost_per_unit=cost_rates.cloud_cpu_cost_per_unit,
        cloud_memory_cost_per_mb_second=(
            cost_rates.cloud_memory_cost_per_mb_second
        ),
        cold_start_cost_per_ms=cost_rates.cold_start_cost_per_ms,
        input_size_mb_per_request=input_size_mb_per_request,
        slot_seconds=slot_seconds,
    )

    # 副本数对应的规划器只在场景工厂中建立并交给共享快层执行器。
    # 环境不再根据动作自行选择规划器，从而保证训练与规则仿真语义一致。
    fast_slot_executor = FastSlotExecutor(
        topology=topology,
        network=network,
        functions=functions,
        sfc=sfc,
        replica_planners={
            1: SingleReplicaPlanner(),
            2: build_reliability_aware_replica_planner(
                config,
                replica_count=2,
            ),
            3: build_reliability_aware_replica_planner(
                config,
                replica_count=3,
            ),
        },
        constraint_auditor=constraint_auditor,
        fast_optimizer=fast_optimizer,
        input_size_mb_per_request=input_size_mb_per_request,
        slot_seconds=slot_seconds,
        handover_hot_window_s=float(
            config["two_timescale"]["handover_hot_window_s"]
        ),
        failover_delay_ms_per_function=float(
            config["runtime_failure"]["failover_delay_ms_per_function"]
        ),
        cost_rates=cost_rates,
        return_result_to_source=True,
    )

    state_encoder = RLStateEncoder(
        trackside_node_ids=tuple(
            site.node.node_id for site in topology.sites
        ),
        cloud_node_id=topology.cloud_node.node_id,
        functions=functions,
        sfc=sfc,
        input_size_mb_per_request=input_size_mb_per_request,
        maximum_cpu_cycles_per_request=max(
            function.cpu_cycles_per_request for function in functions
        ),
        maximum_function_memory_mb=max(
            function.memory_mb for function in functions
        ),
        maximum_execution_time_ms=max(
            function.warm_exec_time_ms for function in functions
        ),
        maximum_cold_start_time_ms=max(
            function.cold_start_time_ms for function in functions
        ),
        maximum_deadline_ms=sfc.deadline_ms,
        maximum_input_size_mb=normalization_config["maximum_input_size_mb"],
        maximum_replica_count=int(config["serverless"]["max_replicas"]),
    )

    def failure_process_builder(seed: int):
        return build_windowed_markov_failure_process(
            config=config,
            topology=topology,
            random_seed=seed,
        )

    return SlowTimescaleRLEnvironment(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=functions,
        sfc=sfc,
        failure_process_builder=failure_process_builder,
        failure_risk_provider=build_windowed_failure_risk_provider(config),
        slow_period_slots=int(
            config["two_timescale"]["slow_period_slots"]
        ),
        slot_seconds=slot_seconds,
        state_encoder=state_encoder,
        workload_predictor=HistoricalWorkloadPredictor(
            lookback_slots=int(prediction_config["lookback_slots"]),
            baseline_request_rate=prediction_config["baseline_request_rate"],
        ),
        fast_slot_executor=fast_slot_executor,
        maximum_request_rate=normalization_config["maximum_request_rate"],
        maximum_network_delay_ms=(
            normalization_config["maximum_network_delay_ms"]
        ),
        maximum_window_cost=cost_config["maximum_window_cost"],
        minimum_distinct_fault_domains=int(
            config["reliability"]["minimum_distinct_fault_domains"]
        ),
        default_seed=int(config["rl_environment"]["default_seed"]),
    )
