"""
run_two_timescale_comparison.py

比较五种SFC可靠性保障策略：

1. 固定单副本；
2. 固定跨故障域冷备；
3. 固定跨故障域全热备；
4. 固定轨迹感知动态主备；
5. 规则式双时间尺度控制。

所有策略使用完全相同的：

1. 列车轨迹；
2. 请求轨迹；
3. 可控基础设施故障；
4. SFC参数；
5. 成本权重。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_process import (
    ScriptedFailureProcess,
)
from src.failure_risk_prediction import (
    build_windowed_failure_risk_provider,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.reliability import (
    build_fault_domain_reliability_model,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.two_timescale_control import (
    FixedModeTwoTimescaleController,
    StandbyMode,
    build_rule_based_two_timescale_controller,
)
from src.two_timescale_simulator import (
    TwoTimescaleCostWeights,
    TwoTimescaleRunResult,
    TwoTimescaleRuntimeSimulator,
)
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[
    ServerlessFunction
]:
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


def build_demo_sfc() -> SFCType:
    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_failure_process(
    config: dict,
    topology,
) -> ScriptedFailureProcess:
    """
    复用第十阶段中的可控故障轨迹。
    """

    down_nodes_by_slot: dict[
        int,
        set[int],
    ] = {}

    for item in config[
        "standby_mode_demo"
    ]["scripted_node_failures"]:
        down_nodes_by_slot.setdefault(
            int(item["time_slot"]),
            set(),
        ).add(
            int(item["node_id"])
        )

    down_domains_by_slot: dict[
        int,
        set[int],
    ] = {}

    for item in config[
        "standby_mode_demo"
    ]["scripted_domain_failures"]:
        down_domains_by_slot.setdefault(
            int(item["time_slot"]),
            set(),
        ).add(
            int(item["domain_id"])
        )

    return ScriptedFailureProcess(
        topology=topology,
        down_domains_by_slot=(
            down_domains_by_slot
        ),
        down_nodes_by_slot=(
            down_nodes_by_slot
        ),
    )


def build_cost_weights(
    config: dict,
) -> TwoTimescaleCostWeights:
    cost_config = config[
        "two_timescale_cost"
    ]

    return TwoTimescaleCostWeights(
        delay_cost_per_ms=float(
            cost_config[
                "delay_cost_per_ms"
            ]
        ),
        memory_cost_per_mb_second=float(
            cost_config[
                "memory_cost_per_mb_second"
            ]
        ),
        cold_start_cost_per_ms=float(
            cost_config[
                "cold_start_cost_per_ms"
            ]
        ),
        sla_violation_penalty_per_batch=float(
            cost_config[
                "sla_violation_penalty_per_batch"
            ]
        ),
        slow_decision_cost=float(
            cost_config[
                "slow_decision_cost"
            ]
        ),
        replica_reconfiguration_cost_per_function=float(
            cost_config[
                "replica_reconfiguration_cost_per_function"
            ]
        ),
    )


def build_simulator(
    config: dict,
    controller,
) -> TwoTimescaleRuntimeSimulator:
    topology = build_linear_topology(config)

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=(
            config["train"][
                "initial_position_m"
            ]
        ),
        speed_mps=(
            config["train"]["speed_mps"]
        ),
        slot_seconds=(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
    )

    workload = DeterministicWorkload(
        request_trace=(
            config[
                "integrated_simulation"
            ]["request_trace"]
        ),
        repeat=(
            config[
                "integrated_simulation"
            ]["repeat_request_trace"]
        ),
    )

    return TwoTimescaleRuntimeSimulator(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        controller=controller,
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process=(
            build_failure_process(
                config=config,
                topology=topology,
            )
        ),
        failure_risk_provider=(
            build_windowed_failure_risk_provider(
                config
            )
        ),
        reliability_model=(
            build_fault_domain_reliability_model(
                config=config,
                topology=topology,
            )
        ),
        prediction_horizon_slots=int(
            config["two_timescale"][
                "slow_period_slots"
            ]
        ),
        input_size_mb_per_request=float(
            config[
                "integrated_simulation"
            ][
                "input_size_mb_per_request"
            ]
        ),
        slot_seconds=float(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
        failover_delay_ms_per_function=float(
            config["runtime_failure"][
                "failover_delay_ms_per_function"
            ]
        ),
        cost_weights=build_cost_weights(
            config
        ),
        return_result_to_source=True,
    )


def print_result(
    scenario_name: str,
    result: TwoTimescaleRunResult,
) -> None:
    summary = result.summary

    print("\n" + "=" * 72)
    print(f"方案：{scenario_name}")
    print("=" * 72)

    print(
        f"  请求成功率："
        f"{summary.request_success_rate:.6f}"
    )

    print(
        f"  失败请求数："
        f"{summary.failed_requests}"
    )

    print(
        f"  SLA违反率："
        f"{summary.sla_violation_rate:.6f}"
    )

    print(
        f"  故障接管批次数："
        f"{summary.failover_batches}"
    )

    print(
        f"  冷启动函数阶段数："
        f"{summary.cold_start_function_stages}"
    )

    print(
        f"  冷启动总时延："
        f"{summary.total_cold_start_delay_ms:.3f} ms"
    )

    print(
        f"  成功批次平均时延："
        f"{summary.average_successful_batch_delay_ms:.3f} ms"
    )

    print(
        f"  成功批次P95时延："
        f"{summary.p95_successful_batch_delay_ms:.3f} ms"
    )

    print(
        f"  平均活动内存："
        f"{summary.average_active_memory_mb:.3f} MB"
    )

    print(
        f"  峰值活动内存："
        f"{summary.peak_active_memory_mb:.3f} MB"
    )

    print(
        f"  慢决策执行次数："
        f"{summary.slow_decision_updates}"
    )

    print(
        f"  慢模式切换次数："
        f"{summary.slow_mode_switches}"
    )

    print(
        f"  副本重配置函数阶段数："
        f"{summary.replica_reconfiguration_function_stages}"
    )

    print(
        f"  用户时延成本："
        f"{summary.total_request_delay_cost:.3f}"
    )

    print(
        f"  内存成本："
        f"{summary.total_memory_cost:.3f}"
    )

    print(
        f"  冷启动资源成本："
        f"{summary.total_cold_start_cost:.3f}"
    )

    print(
        f"  SLA惩罚："
        f"{summary.total_sla_penalty:.3f}"
    )

    print(
        f"  慢控制与重配置成本："
        f"{summary.total_slow_control_cost:.3f}"
    )

    print(
        f"  系统综合成本："
        f"{summary.total_system_cost:.3f}"
    )


def print_two_timescale_events(
    result: TwoTimescaleRunResult,
) -> None:
    """
    打印双时间尺度方案的关键慢动作和故障事件。
    """

    print("\n" + "=" * 72)
    print("双时间尺度方案关键事件")
    print("=" * 72)

    event_number = 0

    for record in result.records:
        important = (
            record.slow_decision_updated
            or record.slow_mode_changed
            or record.backup_activation_triggered
            or len(record.failover_function_ids) > 0
            or record.request_success is False
        )

        if not important:
            continue

        event_number += 1

        print(f"\n事件 {event_number}")
        print("-" * 72)

        print(
            f"  时隙与位置："
            f"{record.time_slot}，"
            f"{record.position_m:.2f} m"
        )

        print(
            f"  当前接入："
            f"MEC-{record.serving_mec + 1}"
        )

        print(
            f"  预测负载："
            f"{record.predicted_request_rate:.3f}"
        )

        print(
            f"  预测故障风险："
            f"{record.predicted_failure_risk:.3f}"
        )

        print(
            f"  当前慢模式："
            f"{record.slow_mode}"
        )

        print(
            f"  是否更新慢决策："
            f"{record.slow_decision_updated}"
        )

        print(
            f"  是否发生模式切换："
            f"{record.slow_mode_changed}"
        )

        print(
            f"  是否临时激活备用："
            f"{record.backup_activation_triggered}"
        )

        print(
            f"  故障接管函数："
            f"{list(record.failover_function_ids)}"
        )

        print(
            f"  冷启动函数："
            f"{list(record.cold_start_function_ids)}"
        )

        print(
            f"  请求状态："
            f"{record.request_success}"
        )

        if record.end_to_end_delay_ms is not None:
            print(
                f"  端到端时延："
                f"{record.end_to_end_delay_ms:.3f} ms"
            )

        print(
            f"  慢决策原因："
            f"{record.slow_reason}"
        )


def save_summary_csv(
    results: list[
        tuple[str, TwoTimescaleRunResult]
    ],
    output_path: Path,
) -> None:
    fieldnames = [
        "scenario",
        "request_success_rate",
        "failed_requests",
        "sla_violation_rate",
        "failover_batches",
        "cold_start_function_stages",
        "total_cold_start_delay_ms",
        "average_successful_batch_delay_ms",
        "p95_successful_batch_delay_ms",
        "average_active_memory_mb",
        "peak_active_memory_mb",
        "slow_decision_updates",
        "slow_mode_switches",
        "replica_reconfiguration_function_stages",
        "total_request_delay_cost",
        "total_memory_cost",
        "total_cold_start_cost",
        "total_sla_penalty",
        "total_slow_control_cost",
        "total_system_cost",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for scenario_name, result in results:
            summary = result.summary

            writer.writerow(
                {
                    "scenario": scenario_name,
                    **{
                        field_name: getattr(
                            summary,
                            field_name,
                        )
                        for field_name
                        in fieldnames
                        if field_name != "scenario"
                    },
                }
            )


def plot_metric(
    results: list[
        tuple[str, TwoTimescaleRunResult]
    ],
    values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
    rate_metric: bool = False,
) -> None:
    scenario_names = [
        scenario_name
        for scenario_name, _ in results
    ]

    figure, axis = plt.subplots(
        figsize=(11, 5)
    )

    axis.bar(
        scenario_names,
        values,
    )

    axis.set_title(title)
    axis.set_xlabel("Control Strategy")
    axis.set_ylabel(y_label)

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    if rate_metric:
        axis.set_ylim(0.0, 1.05)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
    )

    plt.close(figure)


def main() -> None:
    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    handover_window = float(
        config["two_timescale"][
            "handover_hot_window_s"
        ]
    )

    scenarios = [
        (
            "fixed_single",
            FixedModeTwoTimescaleController(
                standby_mode=StandbyMode.SINGLE,
                handover_hot_window_s=(
                    handover_window
                ),
            ),
        ),
        (
            "fixed_cold",
            FixedModeTwoTimescaleController(
                standby_mode=StandbyMode.COLD,
                handover_hot_window_s=(
                    handover_window
                ),
                enable_handover_activation=False,
            ),
        ),
        (
            "fixed_hot",
            FixedModeTwoTimescaleController(
                standby_mode=StandbyMode.HOT,
                handover_hot_window_s=(
                    handover_window
                ),
            ),
        ),
        (
            "fixed_dynamic",
            FixedModeTwoTimescaleController(
                standby_mode=StandbyMode.COLD,
                handover_hot_window_s=(
                    handover_window
                ),
                enable_handover_activation=True,
            ),
        ),
        (
            "two_timescale",
            build_rule_based_two_timescale_controller(
                config
            ),
        ),
    ]

    results: list[
        tuple[str, TwoTimescaleRunResult]
    ] = []

    for scenario_name, controller in scenarios:
        simulator = build_simulator(
            config=config,
            controller=controller,
        )

        result = simulator.run()

        results.append(
            (
                scenario_name,
                result,
            )
        )

        print_result(
            scenario_name=scenario_name,
            result=result,
        )

    print_two_timescale_events(
        result=results[-1][1]
    )

    tables_dir = (
        project_root / "results" / "tables"
    )

    figures_dir = (
        project_root / "results" / "figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        tables_dir
        / "two_timescale_comparison.csv"
    )

    success_path = (
        figures_dir
        / "two_timescale_success_rate.png"
    )

    sla_path = (
        figures_dir
        / "two_timescale_sla_violation.png"
    )

    memory_path = (
        figures_dir
        / "two_timescale_average_memory.png"
    )

    cost_path = (
        figures_dir
        / "two_timescale_total_cost.png"
    )

    save_summary_csv(
        results=results,
        output_path=summary_path,
    )

    plot_metric(
        results=results,
        values=[
            result.summary.request_success_rate
            for _, result in results
        ],
        title=(
            "Two-Timescale Control: "
            "Request Success Rate"
        ),
        y_label="Request Success Rate",
        output_path=success_path,
        rate_metric=True,
    )

    plot_metric(
        results=results,
        values=[
            result.summary.sla_violation_rate
            for _, result in results
        ],
        title=(
            "Two-Timescale Control: "
            "SLA Violation Rate"
        ),
        y_label="SLA Violation Rate",
        output_path=sla_path,
        rate_metric=True,
    )

    plot_metric(
        results=results,
        values=[
            result.summary.average_active_memory_mb
            for _, result in results
        ],
        title=(
            "Two-Timescale Control: "
            "Average Active Memory"
        ),
        y_label="Average Active Memory (MB)",
        output_path=memory_path,
    )

    plot_metric(
        results=results,
        values=[
            result.summary.total_system_cost
            for _, result in results
        ],
        title=(
            "Two-Timescale Control: "
            "Total System Cost"
        ),
        y_label="Normalized Total Cost",
        output_path=cost_path,
    )

    print("\n" + "=" * 72)
    print("双时间尺度正式对比结果已保存")
    print("=" * 72)
    print(f"汇总表：{summary_path}")
    print(f"请求成功率图：{success_path}")
    print(f"SLA违反率图：{sla_path}")
    print(f"平均内存图：{memory_path}")
    print(f"系统综合成本图：{cost_path}")


if __name__ == "__main__":
    main()
