"""
run_prediction_error_demo.py

演示不同轨迹预测误差下，
提前预热策略在什么时隙启动预热。

本程序只观察“预热决策时机”，
暂时不计算预热浪费和资源成本。

比较四种情况：

1. 理想无误差预测；
2. 提前4秒预测；
3. 滞后4秒预测；
4. 下一MEC预测多偏移1跳。
"""

from pathlib import Path

from src.config import load_config
from src.entities import (
    ServicePriority,
    SFCType,
)
from src.mobility import TrainMobilityModel
from src.prewarming import (
    TrajectoryAwarePrewarmingPolicy,
)
from src.topology import build_linear_topology
from src.trajectory_prediction import (
    BiasedTrajectoryPredictor,
    PerfectTrajectoryPredictor,
)


def build_demo_sfc() -> SFCType:
    """
    创建演示使用的三函数 SFC。

    这个程序只关心预热目标，
    暂时不需要具体函数执行参数。
    """

    return SFCType(
        sfc_id=0,
        name="列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def print_policy_decisions(
    policy_name: str,
    policy: TrajectoryAwarePrewarmingPolicy,
    config: dict,
) -> None:
    """
    打印一种预测策略首次预热各目标 MEC 的时隙。
    """

    topology = build_linear_topology(config)

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=(
            config["train"]["initial_position_m"]
        ),
        speed_mps=config["train"]["speed_mps"],
        slot_seconds=(
            config["simulation"]["fast_slot_seconds"]
        ),
    )

    sfc = build_demo_sfc()

    train_state = mobility_model.reset()

    # 一个目标 MEC 可能连续多个时隙都处于预热窗口。
    #
    # started_targets 用于只打印第一次触发时刻。
    started_targets: set[int] = set()

    print("\n" + "=" * 72)
    print(f"预测场景：{policy_name}")
    print("=" * 72)

    while True:
        target_node_ids = policy.target_node_ids(
            train_state=train_state,
            sfc=sfc,
            topology=topology,
        )

        for target_node_id in target_node_ids:
            if target_node_id in started_targets:
                continue

            started_targets.add(target_node_id)

            prediction = policy.predictor.predict(
                train_state=train_state,
                topology=topology,
            )

            print(
                f"时隙 {train_state.time_slot:3d}："
                f"位置 {train_state.position_m:7.2f} m，"
                f"当前 MEC-{train_state.serving_mec + 1}，"
                f"预测预热 MEC-{target_node_id + 1}"
            )

            print(
                f"    真实剩余驻留时间："
                f"{train_state.remaining_dwell_time_s:.2f} s"
            )

            print(
                f"    预测剩余驻留时间："
                f"{prediction.predicted_remaining_dwell_time_s:.2f} s"
            )

        if mobility_model.finished:
            break

        train_state = mobility_model.step()

    if len(started_targets) == 0:
        print("整个行程中没有触发提前预热。")


def main() -> None:
    """
    程序主入口。
    """

    project_root = Path(__file__).resolve().parent

    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    lead_time_s = config["prewarming_demo"]["lead_time_s"]

    scenarios = [
        (
            "理想无误差预测",
            PerfectTrajectoryPredictor(),
        ),
        (
            "提前4秒预测",
            BiasedTrajectoryPredictor(
                time_bias_s=-4.0,
            ),
        ),
        (
            "滞后4秒预测",
            BiasedTrajectoryPredictor(
                time_bias_s=4.0,
            ),
        ),
        (
            "目标MEC向前偏移1跳",
            BiasedTrajectoryPredictor(
                time_bias_s=0.0,
                next_mec_hop_offset=1,
            ),
        ),
    ]

    print("=" * 72)
    print("轨迹预测误差对提前预热决策的影响")
    print(f"预热时间阈值：{lead_time_s:.1f} 秒")
    print("=" * 72)

    for scenario_name, predictor in scenarios:
        policy = TrajectoryAwarePrewarmingPolicy(
            lead_time_s=lead_time_s,
            predictor=predictor,
        )

        print_policy_decisions(
            policy_name=scenario_name,
            policy=policy,
            config=config,
        )


if __name__ == "__main__":
    main()