"""
run_reliability_demo.py

比较三种 SFC 副本部署方式：

1. 单副本部署；
2. 同故障域双副本；
3. 跨故障域主备副本。

程序输出：

1. 每个函数的可用率；
2. SFC 阶段独立近似可靠性；
3. 考虑共享故障后的精确可靠性；
4. 故障域隔离约束；
5. 是否满足 SFC 可靠性要求。

同时生成 CSV 和可靠性柱状图。
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
    TrainState,
)
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.reliability import (
    SFCReliabilityResult,
    build_fault_domain_reliability_model,
)
from src.topology import build_linear_topology


def build_demo_sfc() -> SFCType:
    """
    创建高可靠列车故障诊断 SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_demo_train_state() -> TrainState:
    """
    创建列车位于 MEC-1 服务区的状态。
    """

    return TrainState(
        time_slot=0,
        position_m=0.0,
        speed_mps=69.44,
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=14.4,
    )


def print_topology_fault_domains(
    topology,
) -> None:
    """
    打印每个 MEC 所属故障域。
    """

    print("\n" + "=" * 72)
    print("轨旁 MEC 与故障域")
    print("=" * 72)

    for site in topology.sites:
        print(
            f"  {site.node.name}："
            f"位置 {site.position_m:.0f} m，"
            f"故障域 {site.node.fault_domain}，"
            f"节点局部可用率 "
            f"{site.node.reliability:.4f}"
        )


def print_plan_result(
    scenario_name: str,
    replica_nodes: tuple[int, ...],
    result: SFCReliabilityResult,
) -> None:
    """
    分块打印一种副本部署方案。
    """

    replica_text = "、".join(
        f"MEC-{node_id + 1}"
        for node_id in replica_nodes
    )

    print("\n" + "=" * 72)
    print(f"场景：{scenario_name}")
    print("=" * 72)

    print(f"  副本节点：{replica_text}")

    domain_text = "、".join(
        str(domain_id)
        for domain_id
        in result.function_results[0].fault_domain_ids
    )

    print(f"  覆盖故障域：{domain_text}")

    for function_result in result.function_results:
        print(
            f"  函数 {function_result.function_id} "
            f"可用率："
            f"{function_result.availability:.9f}"
        )

    print(
        f"  函数阶段独立近似可靠性："
        f"{result.approximate_stage_product_availability:.9f}"
    )

    print(
        f"  共享故障精确可靠性："
        f"{result.exact_shared_failure_availability:.9f}"
    )

    print(
        f"  SFC 可靠性要求："
        f"{result.reliability_target:.6f}"
    )

    print(
        f"  故障域隔离要求："
        f"{'满足' if result.fault_domain_diversity_met else '不满足'}"
    )

    print(
        f"  最终可靠性约束："
        f"{'满足' if result.target_met else '不满足'}"
    )


def save_summary_csv(
    scenario_results: list[
        tuple[
            str,
            tuple[int, ...],
            SFCReliabilityResult,
        ]
    ],
    output_path: Path,
) -> None:
    """
    保存可靠性对比结果。
    """

    fieldnames = [
        "scenario",
        "replica_node_ids",
        "fault_domain_ids",
        "function_availability",
        "approximate_stage_product_availability",
        "exact_shared_failure_availability",
        "reliability_target",
        "fault_domain_diversity_met",
        "target_met",
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

        for (
            scenario_name,
            replica_nodes,
            result,
        ) in scenario_results:
            writer.writerow(
                {
                    "scenario": scenario_name,
                    "replica_node_ids": list(
                        replica_nodes
                    ),
                    "fault_domain_ids": list(
                        result.function_results[
                            0
                        ].fault_domain_ids
                    ),
                    "function_availability": (
                        result.function_results[
                            0
                        ].availability
                    ),
                    "approximate_stage_product_availability": (
                        result
                        .approximate_stage_product_availability
                    ),
                    "exact_shared_failure_availability": (
                        result
                        .exact_shared_failure_availability
                    ),
                    "reliability_target": (
                        result.reliability_target
                    ),
                    "fault_domain_diversity_met": (
                        result
                        .fault_domain_diversity_met
                    ),
                    "target_met": (
                        result.target_met
                    ),
                }
            )


def plot_reliability(
    scenario_results: list[
        tuple[
            str,
            tuple[int, ...],
            SFCReliabilityResult,
        ]
    ],
    output_path: Path,
) -> None:
    """
    绘制三种方案的精确 SFC 可靠性。
    """

    scenario_names = [
        scenario_name
        for scenario_name, _, _
        in scenario_results
    ]

    exact_availabilities = [
        result.exact_shared_failure_availability
        for _, _, result
        in scenario_results
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        scenario_names,
        exact_availabilities,
    )

    target = (
        scenario_results[0][2]
        .reliability_target
    )

    axis.axhline(
        y=target,
        linestyle="--",
        label=f"Reliability target = {target:.3f}",
    )

    # 将纵轴限制在较高可靠性区间，
    # 方便观察不同方案的差距。
    axis.set_ylim(0.95, 1.0005)

    axis.set_title(
        "Fault-Domain-Aware SFC Reliability"
    )

    axis.set_xlabel("Replica Placement")
    axis.set_ylabel("Exact SFC Availability")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """
    程序主入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    topology = build_linear_topology(config)

    reliability_model = (
        build_fault_domain_reliability_model(
            config=config,
            topology=topology,
        )
    )

    replica_planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    sfc = build_demo_sfc()
    train_state = build_demo_train_state()

    # --------------------------------------------------------
    # 方案1：所有函数只有 MEC-1 上的单副本。
    # --------------------------------------------------------

    single_replica_nodes = (0,)

    single_plan = {
        function_id: single_replica_nodes
        for function_id in sfc.function_ids
    }

    single_result = reliability_model.evaluate_sfc(
        sfc=sfc,
        function_replica_node_ids=single_plan,
    )

    # --------------------------------------------------------
    # 方案2：MEC-1 和 MEC-2 双副本。
    #
    # 两个节点都属于故障域0。
    # --------------------------------------------------------

    same_domain_nodes = (0, 1)

    same_domain_plan = {
        function_id: same_domain_nodes
        for function_id in sfc.function_ids
    }

    same_domain_result = (
        reliability_model.evaluate_sfc(
            sfc=sfc,
            function_replica_node_ids=(
                same_domain_plan
            ),
        )
    )

    # --------------------------------------------------------
    # 方案3：由故障域感知规划器自动选择主备节点。
    #
    # 当前应选择 MEC-1 和 MEC-3。
    # --------------------------------------------------------

    cross_domain_plan = replica_planner.plan(
        sfc=sfc,
        train_state=train_state,
        topology=topology,
    )

    cross_domain_result = (
        reliability_model.evaluate_sfc(
            sfc=sfc,
            function_replica_node_ids=(
                cross_domain_plan
                .function_replica_node_ids
            ),
        )
    )

    scenario_results = [
        (
            "single_replica",
            single_replica_nodes,
            single_result,
        ),
        (
            "same_domain_replicas",
            same_domain_nodes,
            same_domain_result,
        ),
        (
            "cross_domain_replicas",
            cross_domain_plan
            .shared_replica_node_ids,
            cross_domain_result,
        ),
    ]

    print_topology_fault_domains(
        topology
    )

    for (
        scenario_name,
        replica_nodes,
        result,
    ) in scenario_results:
        print_plan_result(
            scenario_name=scenario_name,
            replica_nodes=replica_nodes,
            result=result,
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
        / "reliability_plan_comparison.csv"
    )

    figure_path = (
        figures_dir
        / "reliability_plan_comparison.png"
    )

    save_summary_csv(
        scenario_results=scenario_results,
        output_path=summary_path,
    )

    plot_reliability(
        scenario_results=scenario_results,
        output_path=figure_path,
    )

    print("\n" + "=" * 72)
    print("实验结果已保存")
    print("=" * 72)
    print(f"可靠性汇总表：{summary_path}")
    print(f"可靠性对比图：{figure_path}")


if __name__ == "__main__":
    main()