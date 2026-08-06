"""
run_two_timescale_decision_demo.py

演示双时间尺度控制器如何工作。

慢时间尺度：

1. 每10个快时隙更新一次；
2. MEC切换时强制更新；
3. 根据可靠性目标、预测故障风险和请求负载，
   决定单副本、冷备或全热备。

快时间尺度：

1. 每个时隙判断是否接近MEC切换；
2. 冷备模式临近切换时临时激活备用实例；
3. 主节点失效时选择备用节点；
4. 判断备用接管是否需要冷启动。
"""

from pathlib import Path

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
)
from src.mobility import TrainMobilityModel
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.topology import build_linear_topology
from src.two_timescale_control import (
    FastTimescaleState,
    SlowTimescaleState,
    StandbyMode,
    build_rule_based_two_timescale_controller,
)
from src.workload import DeterministicWorkload


def build_demo_sfc() -> SFCType:
    """
    创建双时间尺度演示SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def predicted_failure_risk(
    time_slot: int,
) -> float:
    """
    构造演示使用的预测故障风险。

    时隙30到49：
        预测故障风险较高。

    其他时隙：
        预测风险较低。
    """

    if 30 <= time_slot <= 49:
        return 0.20

    return 0.02


def unavailable_nodes(
    time_slot: int,
) -> set[int]:
    """
    构造两个主节点故障时隙。

    时隙17：
        MEC-2局部失效。

    时隙44：
        MEC-3局部失效。
    """

    if time_slot == 17:
        return {1}

    if time_slot == 44:
        return {2}

    return set()


def main() -> None:
    """
    程序入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    topology = build_linear_topology(config)

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=(
            config["train"]["initial_position_m"]
        ),
        speed_mps=config["train"]["speed_mps"],
        slot_seconds=(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
    )

    workload = DeterministicWorkload(
        request_trace=(
            config["integrated_simulation"][
                "request_trace"
            ]
        ),
        repeat=(
            config["integrated_simulation"][
                "repeat_request_trace"
            ]
        ),
    )

    sfc = build_demo_sfc()

    controller = (
        build_rule_based_two_timescale_controller(
            config
        )
    )

    single_planner = SingleReplicaPlanner()

    redundant_planner = (
        build_reliability_aware_replica_planner(
            config
        )
    )

    controller.reset()
    train_state = mobility_model.reset()

    all_node_ids = {
        site.node.node_id
        for site in topology.sites
    }

    previous_slow_decision_slot: int | None = None

    print("=" * 72)
    print("双时间尺度 Serverless SFC 决策演示")
    print("=" * 72)

    while True:
        request_count = workload.request_count(
            train_state.time_slot
        )

        failure_risk = predicted_failure_risk(
            train_state.time_slot
        )

        slow_state = SlowTimescaleState(
            time_slot=train_state.time_slot,
            serving_mec=train_state.serving_mec,
            next_mec=train_state.next_mec,
            predicted_request_rate=float(
                request_count
            ),
            predicted_failure_risk=(
                failure_risk
            ),
            reliability_target=(
                sfc.reliability_target
            ),
        )

        slow_decision = (
            controller.get_slow_decision(
                slow_state
            )
        )

        # 慢时间尺度启用冗余时，
        # 使用跨故障域副本规划器。
        if slow_decision.use_redundancy:
            planner = redundant_planner
        else:
            planner = single_planner

        placement_plan = planner.plan(
            sfc=sfc,
            train_state=train_state,
            topology=topology,
        )

        candidate_map = {
            function_id: tuple(node_ids)
            for function_id, node_ids
            in placement_plan
            .function_replica_node_ids
            .items()
        }

        down_nodes = unavailable_nodes(
            train_state.time_slot
        )

        operational_nodes = (
            all_node_ids - down_nodes
        )

        fast_state = FastTimescaleState(
            time_slot=train_state.time_slot,
            serving_mec=train_state.serving_mec,
            remaining_dwell_time_s=(
                train_state
                .remaining_dwell_time_s
            ),
            request_count=request_count,
            function_ids=tuple(
                sfc.function_ids
            ),
            candidate_node_ids=candidate_map,
            operational_node_ids=frozenset(
                operational_nodes
            ),
        )

        fast_decision = (
            controller.get_fast_decision(
                state=fast_state,
                slow_decision=slow_decision,
            )
        )

        # ----------------------------------------------------
        # 打印新的慢时间尺度决策。
        # ----------------------------------------------------

        if (
            slow_decision.decision_slot
            != previous_slow_decision_slot
        ):
            previous_slow_decision_slot = (
                slow_decision.decision_slot
            )

            print("\n" + "=" * 72)
            print(
                f"慢时间尺度决策："
                f"时隙{slow_decision.decision_slot}"
            )
            print("=" * 72)

            print(
                f"  当前接入："
                f"MEC-{train_state.serving_mec + 1}"
            )

            print(
                f"  预测故障风险："
                f"{failure_risk:.3f}"
            )

            print(
                f"  预测请求负载："
                f"{request_count:.3f}"
            )

            print(
                f"  是否启用冗余："
                f"{slow_decision.use_redundancy}"
            )

            print(
                f"  每函数副本数："
                f"{slow_decision.replica_count}"
            )

            print(
                f"  基础主备模式："
                f"{slow_decision.standby_mode.value}"
            )

            print(
                f"  正常有效期："
                f"{slow_decision.decision_slot}"
                f"至"
                f"{slow_decision.valid_until_slot}"
            )

            print(
                f"  决策原因："
                f"{slow_decision.reason}"
            )

        # ----------------------------------------------------
        # 打印关键快时间尺度动作。
        # ----------------------------------------------------

        important_fast_event = (
            fast_decision
            .backup_activation_triggered
            or len(
                fast_decision
                .failover_function_ids
            ) > 0
            or fast_decision.request_success is False
        )

        if important_fast_event:
            print("\n  快时间尺度关键动作")
            print("  " + "-" * 68)

            print(
                f"  时隙与位置："
                f"{train_state.time_slot}，"
                f"{train_state.position_m:.2f} m"
            )

            print(
                f"  剩余驻留时间："
                f"{train_state.remaining_dwell_time_s:.3f} s"
            )

            print(
                f"  当前请求数："
                f"{request_count}"
            )

            print(
                f"  失效节点："
                f"{[node_id + 1 for node_id in down_nodes]}"
            )

            print(
                f"  是否临时激活备用："
                f"{fast_decision.backup_activation_triggered}"
            )

            print(
                f"  故障接管函数："
                f"{list(fast_decision.failover_function_ids)}"
            )

            print(
                f"  需要冷启动的函数："
                f"{list(fast_decision.cold_start_function_ids)}"
            )

            execution_node_names = [
                node_id + 1
                for node_id
                in fast_decision.selected_execution_node_ids
            ]

            print(
                f"  实际执行节点："
                f"{execution_node_names}"
            )

            print(
                f"  请求是否成功："
                f"{fast_decision.request_success}"
            )

        if mobility_model.finished:
            break

        train_state = mobility_model.step()

    print("\n" + "=" * 72)
    print("双时间尺度决策演示完成")
    print("=" * 72)


if __name__ == "__main__":
    main()