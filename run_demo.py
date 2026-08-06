"""
run_demo.py

第一阶段演示程序。

运行本文件后，它会：

1. 读取 configs/debug.yaml；
2. 创建 5 个轨旁 MEC；
3. 创建 3 个 Serverless 函数；
4. 创建一条故障诊断 SFC；
5. 创建一批请求；
6. 创建一个列车状态；
7. 将这些信息打印到屏幕。
"""

from pathlib import Path

from src.config import load_config
from src.entities import (
    EdgeNode,
    NodeType,
    RequestBatch,
    ServerlessFunction,
    ServicePriority,
    SFCType,
    TrainState,
)


def build_trackside_mecs(config: dict) -> list[EdgeNode]:
    """
    根据配置文件创建多个轨旁 MEC。

    Parameters
    ----------
    config:
        从 debug.yaml 读取的完整配置字典。

    Returns
    -------
    list[EdgeNode]:
        轨旁 MEC 节点列表。
    """

    mec_count = config["topology"]["mec_count"]
    cpu_capacity = config["node_resources"]["mec_cpu_capacity"]
    memory_mb = config["node_resources"]["mec_memory_mb"]
    reliability = config["node_resources"]["mec_reliability"]

    nodes: list[EdgeNode] = []

    for index in range(mec_count):
        node = EdgeNode(
            node_id=index,
            name=f"MEC-{index + 1}",
            node_type=NodeType.TRACKSIDE,
            cpu_capacity=cpu_capacity,
            memory_capacity_mb=memory_mb,
            reliability=reliability,

            # 调试阶段规定相邻两个 MEC 属于同一个故障域。
            # MEC-1、MEC-2 属于故障域0；
            # MEC-3、MEC-4 属于故障域1；
            # MEC-5 属于故障域2。
            fault_domain=index // 2,
        )

        nodes.append(node)

    return nodes


def build_functions() -> list[ServerlessFunction]:
    """
    创建一条故障诊断业务所需的三个函数。

    当前数值只用于调试，后续会使用数据集和实测数据更新。
    """

    data_cleaning = ServerlessFunction(
        function_id=0,
        name="数据清洗",
        memory_mb=256.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=80.0,
        warm_exec_time_ms=15.0,
        cold_start_time_ms=300.0,
        output_ratio=0.7,
    )

    feature_extraction = ServerlessFunction(
        function_id=1,
        name="特征提取",
        memory_mb=512.0,
        cpu_cycles_per_request=40.0,
        image_size_mb=150.0,
        warm_exec_time_ms=30.0,
        cold_start_time_ms=500.0,
        output_ratio=0.4,
    )

    anomaly_detection = ServerlessFunction(
        function_id=2,
        name="异常检测",
        memory_mb=768.0,
        cpu_cycles_per_request=60.0,
        image_size_mb=300.0,
        warm_exec_time_ms=50.0,
        cold_start_time_ms=800.0,
        output_ratio=0.1,
    )

    return [
        data_cleaning,
        feature_extraction,
        anomaly_detection,
    ]


def main() -> None:
    """
    程序主入口。

    Python 执行 run_demo.py 时，会从这里开始运行。
    """

    # 获取当前 run_demo.py 所在的项目根目录。
    project_root = Path(__file__).resolve().parent

    # 拼接配置文件的完整路径。
    config_path = project_root / "configs" / "debug.yaml"

    # 读取配置。
    config = load_config(config_path)

    # 根据配置创建轨旁 MEC。
    mec_nodes = build_trackside_mecs(config)

    # 创建 Serverless 函数。
    functions = build_functions()

    # 创建一条故障诊断 SFC。
    fault_diagnosis_sfc = SFCType(
        sfc_id=0,
        name="列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )

    # 创建第0时隙到达的一批请求。
    request_batch = RequestBatch(
        time_slot=0,
        sfc_id=0,
        request_count=10,
        input_size_mb=2.0,
    )

    # 创建第0时隙的列车状态。
    train_state = TrainState(
        time_slot=0,
        position_m=config["train"]["initial_position_m"],
        speed_mps=config["train"]["speed_mps"],
        serving_mec=0,
        next_mec=1,
        remaining_dwell_time_s=20.0,
    )

    print("=" * 60)
    print("轨道边缘 Serverless SFC 项目：基础对象演示")
    print("=" * 60)

    print("\n1. 配置文件读取成功")
    print(f"   快时隙长度：{config['simulation']['fast_slot_seconds']} 秒")
    print(f"   慢帧长度：{config['simulation']['slow_frame_slots']} 个快时隙")
    print(f"   随机种子：{config['simulation']['random_seed']}")

    print("\n2. 轨旁 MEC 节点")
    for node in mec_nodes:
        print(
            f"   {node.name}: "
            f"CPU={node.cpu_capacity}, "
            f"内存={node.memory_capacity_mb} MB, "
            f"可靠性={node.reliability}, "
            f"故障域={node.fault_domain}"
        )

    print("\n3. Serverless 函数")
    for function in functions:
        print(
            f"   函数{function.function_id} {function.name}: "
            f"内存={function.memory_mb} MB, "
            f"温执行={function.warm_exec_time_ms} ms, "
            f"冷启动={function.cold_start_time_ms} ms"
        )

    print("\n4. SFC 信息")
    print(f"   名称：{fault_diagnosis_sfc.name}")
    print(f"   函数执行顺序：{fault_diagnosis_sfc.function_ids}")
    print(f"   最大时延：{fault_diagnosis_sfc.deadline_ms} ms")
    print(f"   可靠性要求：{fault_diagnosis_sfc.reliability_target}")
    print(f"   优先级：{fault_diagnosis_sfc.priority.value}")

    print("\n5. 请求信息")
    print(f"   到达时隙：{request_batch.time_slot}")
    print(f"   请求数量：{request_batch.request_count}")
    print(f"   单请求输入数据量：{request_batch.input_size_mb} MB")

    print("\n6. 列车状态")
    print(f"   位置：{train_state.position_m} m")
    print(f"   速度：{train_state.speed_mps} m/s")
    print(f"   当前 MEC：MEC-{train_state.serving_mec + 1}")
    print(f"   下一 MEC：MEC-{train_state.next_mec + 1}")
    print(
        f"   当前覆盖区剩余驻留时间："
        f"{train_state.remaining_dwell_time_s} s"
    )

    print("\n基础对象构建成功。")
    print("=" * 60)


# 只有直接执行 run_demo.py 时，才会调用 main()。
# 将来其他文件导入 run_demo.py 时，不会自动执行。
if __name__ == "__main__":
    main()