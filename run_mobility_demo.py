"""
run_mobility_demo.py

列车移动和 MEC 切换演示程序。

运行后会显示：

1. 所有轨旁 MEC 的位置；
2. 列车随时间移动的位置；
3. 当前接入 MEC；
4. 下一个 MEC；
5. MEC 切换事件。
"""

from pathlib import Path

from src.config import load_config
from src.mobility import TrainMobilityModel
from src.topology import build_linear_topology


def main() -> None:
    """
    程序主入口。
    """

    # 当前项目根目录。
    project_root = Path(__file__).resolve().parent

    # 读取配置文件。
    config = load_config(
        project_root / "configs" / "debug.yaml"
    )

    # 创建铁路拓扑。
    topology = build_linear_topology(config)

    # 创建列车移动模型。
    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=config["train"]["initial_position_m"],
        speed_mps=config["train"]["speed_mps"],
        slot_seconds=config["simulation"]["fast_slot_seconds"],
    )

    print("=" * 78)
    print("轨道边缘 Serverless SFC：列车移动与 MEC 切换演示")
    print("=" * 78)

    print("\n1. 轨旁 MEC 部署位置")

    for site in topology.sites:
        print(
            f"   {site.node.name}: "
            f"位置={site.position_m:.0f} m, "
            f"覆盖半径={site.coverage_radius_m:.0f} m, "
            f"故障域={site.node.fault_domain}"
        )

    print("\n2. 列车移动过程")
    print(
        "   时隙    位置(m)     当前MEC    下一MEC"
        "    剩余驻留时间(s)    事件"
    )
    print("-" * 78)

    state = mobility_model.reset()
    previous_serving_mec = state.serving_mec

    print(
        f"   {state.time_slot:4d}"
        f"    {state.position_m:8.2f}"
        f"     MEC-{state.serving_mec + 1}"
        f"       MEC-{state.next_mec + 1}"
        f"          {state.remaining_dwell_time_s:8.2f}"
        f"       初始状态"
    )

    # 最多运行 200 个时隙，防止异常情况下死循环。
    while not mobility_model.finished and state.time_slot < 200:
        state = mobility_model.step()

        handover_occurred = (
            state.serving_mec != previous_serving_mec
        )

        # 为避免输出太多内容，只在以下情况打印：
        # 1. 每经过 10 个时隙；
        # 2. 发生 MEC 切换；
        # 3. 到达终点。
        should_print = (
            state.time_slot % 10 == 0
            or handover_occurred
            or mobility_model.finished
        )

        if should_print:
            event = ""

            if handover_occurred:
                event = (
                    f"MEC-{previous_serving_mec + 1}"
                    f" → MEC-{state.serving_mec + 1}"
                )
            elif mobility_model.finished:
                event = "到达线路终点"

            print(
                f"   {state.time_slot:4d}"
                f"    {state.position_m:8.2f}"
                f"     MEC-{state.serving_mec + 1}"
                f"       MEC-{state.next_mec + 1}"
                f"          {state.remaining_dwell_time_s:8.2f}"
                f"       {event}"
            )

        previous_serving_mec = state.serving_mec

    print("-" * 78)
    print(
        f"仿真结束：列车在第 {state.time_slot} 个时隙"
        f"到达 {state.position_m:.2f} 米。"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()