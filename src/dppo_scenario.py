"""从统一 YAML 配置构造 DPPO 场景、维度和双时间尺度执行闭环。"""

from typing import Any

from src.constraint_audit import SlotConstraintAuditor
from src.dppo_action_space import DPPOActionSpace
from src.dppo_intent_adapter import DPPOIntentAdapter
from src.dppo_projection import DPPOProjector, ProjectionResourceDemand
from src.dppo_slow_timescale_env import DPPOSlowTimescaleEnvironment
from src.dppo_state_encoder import DPPOStateEncoder
from src.entities import ServerlessFunction, ServicePriority, SFCType
from src.failure_risk_prediction import build_windowed_failure_risk_provider
from src.fast_optimizer import FastFeasibilityOptimizer
from src.fast_slot_executor import FastSlotExecutor, RuntimeCostRates
from src.mobility import TrainMobilityModel
from src.network import build_hybrid_rail_network
from src.redundancy_placement import build_reliability_aware_replica_planner
from src.reliability import build_fault_domain_reliability_model
from src.risk_aware_failure_process import build_windowed_markov_failure_process
from src.runtime_reliability import SingleReplicaPlanner
from src.scenario_dimensions import ScenarioDimensions
from src.slow_timescale_execution_core import SlowTimescaleExecutionCore
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload
from src.workload_prediction import HistoricalWorkloadPredictor


def _scenario_config(config: dict[str, Any]) -> dict[str, Any]:
    """返回配置化 SFC 场景，并给出可定位的缺失错误。"""

    scenario = config.get("rl_scenario")
    if not isinstance(scenario, dict):
        raise ValueError("配置项 rl_scenario 必须是字典。")
    return scenario


def _require_nonnegative_integer(value: Any, name: str) -> int:
    """拒绝布尔值、小数和字符串，避免编号被静默改变。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} 必须是非负整数。")
    return value


def build_rl_functions(config: dict[str, Any]) -> list[ServerlessFunction]:
    """按照 YAML 列表顺序构造 VNF，不在代码中固定函数数量。"""

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
    """构造 SFC，并确认链顺序与联合动作的 VNF 顺序完全一致。"""

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
        allowed = ", ".join(item.value for item in ServicePriority)
        raise ValueError(f"SFC priority 必须是以下值之一：{allowed}。") from error
    return SFCType(
        sfc_id=_require_nonnegative_integer(sfc_config.get("sfc_id"), "SFC ID"),
        name=str(sfc_config["name"]),
        function_ids=list(chain),
        deadline_ms=float(sfc_config["deadline_ms"]),
        reliability_target=float(sfc_config["reliability_target"]),
        priority=priority,
    )


def _environment_subconfig(
    config: dict[str, Any],
    name: str,
) -> dict[str, float]:
    """读取必需环境子配置，禁止在 Python 中隐藏实验参数。"""

    environment_config = config.get("rl_environment")
    if not isinstance(environment_config, dict):
        raise ValueError("rl_environment 必须是字典。")
    supplied = environment_config.get(name)
    if not isinstance(supplied, dict):
        raise ValueError(f"rl_environment.{name} 必须是字典。")
    return {str(key): float(value) for key, value in supplied.items()}


def build_dppo_scenario(config: dict[str, Any]) -> DPPOSlowTimescaleEnvironment:
    """创建配置驱动的 DPPO 场景及唯一共享快层执行器。

    当前场景对象就是可交互的 DPPO 环境。保留独立的场景命名，是为了让
    教师数据生成和在线训练明确复用同一套拓扑、动作空间与执行核心。
    """

    topology = build_linear_topology(config)
    if topology.cloud_node is None:
        raise ValueError("DPPO 场景要求 topology.include_cloud=true。")
    network = build_hybrid_rail_network(config=config, topology=topology)
    functions = build_rl_functions(config)
    sfc = build_rl_sfc(config)
    dimensions = ScenarioDimensions.from_scenario(topology, functions)
    slot_seconds = float(config["simulation"]["fast_slot_seconds"])
    input_size_mb_per_request = float(
        config["rl_scenario"]["sfc"]["input_size_mb_per_request"]
    )
    normalization = _environment_subconfig(
        config,
        "state_normalization",
    )
    prediction = _environment_subconfig(
        config,
        "workload_prediction",
    )
    cost = _environment_subconfig(
        config,
        "cost_rates",
    )

    reliability_model = build_fault_domain_reliability_model(
        config=config,
        topology=topology,
    )
    auditor = SlotConstraintAuditor(
        functions=functions,
        sfc=sfc,
        topology=topology,
        reliability_model=reliability_model,
    )
    cost_rates = RuntimeCostRates(
        edge_cpu_cost_per_unit=cost["edge_cpu_cost_per_unit"],
        edge_memory_cost_per_mb_second=cost["edge_memory_cost_per_mb_second"],
        cloud_cpu_cost_per_unit=cost["cloud_cpu_cost_per_unit"],
        cloud_memory_cost_per_mb_second=cost["cloud_memory_cost_per_mb_second"],
        cold_start_cost_per_ms=cost["cold_start_cost_per_ms"],
    )
    optimizer = FastFeasibilityOptimizer(
        functions=functions,
        sfc=sfc,
        topology=topology,
        auditor=auditor,
        return_result_to_source=True,
        network=network,
        edge_cpu_cost_per_unit=cost_rates.edge_cpu_cost_per_unit,
        edge_memory_cost_per_mb_second=cost_rates.edge_memory_cost_per_mb_second,
        cloud_cpu_cost_per_unit=cost_rates.cloud_cpu_cost_per_unit,
        cloud_memory_cost_per_mb_second=cost_rates.cloud_memory_cost_per_mb_second,
        cold_start_cost_per_ms=cost_rates.cold_start_cost_per_ms,
        input_size_mb_per_request=input_size_mb_per_request,
        slot_seconds=slot_seconds,
    )
    executor = FastSlotExecutor(
        topology=topology,
        network=network,
        functions=functions,
        sfc=sfc,
        replica_planners={
            1: SingleReplicaPlanner(),
            2: build_reliability_aware_replica_planner(config, replica_count=2),
            3: build_reliability_aware_replica_planner(config, replica_count=3),
        },
        constraint_auditor=auditor,
        fast_optimizer=optimizer,
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
    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=float(config["train"]["initial_position_m"]),
        speed_mps=float(config["train"]["speed_mps"]),
        slot_seconds=slot_seconds,
    )
    workload = DeterministicWorkload(
        request_trace=config["integrated_simulation"]["request_trace"],
        repeat=bool(config["integrated_simulation"]["repeat_request_trace"]),
    )

    def failure_process_builder(seed: int):
        """按 Episode 随机种子重建可复现故障轨迹。"""

        return build_windowed_markov_failure_process(
            config=config,
            topology=topology,
            random_seed=seed,
        )

    core = SlowTimescaleExecutionCore(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=functions,
        sfc=sfc,
        failure_process_builder=failure_process_builder,
        failure_risk_provider=build_windowed_failure_risk_provider(config),
        slow_period_slots=int(config["two_timescale"]["slow_period_slots"]),
        slot_seconds=slot_seconds,
        workload_predictor=HistoricalWorkloadPredictor(
            lookback_slots=int(prediction["lookback_slots"]),
            baseline_request_rate=prediction["baseline_request_rate"],
        ),
        fast_slot_executor=executor,
        maximum_request_rate=normalization["maximum_request_rate"],
        maximum_network_delay_ms=normalization["maximum_network_delay_ms"],
        maximum_input_size_mb=normalization["maximum_input_size_mb"],
        input_size_mb_per_request=input_size_mb_per_request,
        maximum_window_cost=cost["maximum_window_cost"],
        default_seed=int(config["rl_environment"]["default_seed"]),
    )
    action_config = config["dppo"]["action"]
    action_space = DPPOActionSpace(
        dimensions=dimensions,
        maximum_retention_seconds=float(
            action_config["maximum_retention_seconds"]
        ),
        replica_threshold=float(action_config["replica_threshold"]),
    )
    projector = DPPOProjector(
        dimensions=dimensions,
        resource_demands={
            function.function_id: ProjectionResourceDemand(
                cpu=function.cpu_cycles_per_request,
                memory_mb=function.memory_mb,
            )
            for function in functions
        },
        minimum_distinct_fault_domains=int(
            config["reliability"]["minimum_distinct_fault_domains"]
        ),
    )
    return DPPOSlowTimescaleEnvironment(
        dimensions=dimensions,
        action_space=action_space,
        projector=projector,
        intent_adapter=DPPOIntentAdapter(),
        state_encoder=DPPOStateEncoder(dimensions),
        execution_core=core,
    )


def build_dppo_environment(config: dict[str, Any]) -> DPPOSlowTimescaleEnvironment:
    """兼容已有调用名称，并统一委托给唯一的 DPPO 场景工厂。"""

    return build_dppo_scenario(config)
