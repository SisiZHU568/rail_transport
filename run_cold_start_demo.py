"""
run_cold_start_demo.py

演示一个 Serverless 函数容器在多个时隙中的状态变化。

请求轨迹：

    时隙0：2个请求
    时隙1：无请求
    时隙2：无请求
    时隙3：1个请求
    时隙4：无请求
    时隙5：无请求
    时隙6：无请求
    时隙7：3个请求
    时隙8：无请求
    时隙9：1个请求

容器生存窗口设为3个空闲时隙。
"""

from src.cold_start import ContainerLifecycleManager
from src.entities import FunctionInstance, ServerlessFunction


def main() -> None:
    """
    程序主入口。
    """

    # 创建用于演示的数据清洗函数。
    function = ServerlessFunction(
        function_id=0,
        name="数据清洗",
        memory_mb=256.0,
        cpu_cycles_per_request=20.0,
        image_size_mb=80.0,
        warm_exec_time_ms=15.0,
        cold_start_time_ms=300.0,
        output_ratio=0.7,
    )

    # 初始状态下，MEC-1 上不存在该函数容器。
    instance = FunctionInstance(
        node_id=0,
        function_id=0,
    )

    # 连续空闲3个时隙后销毁容器。
    manager = ContainerLifecycleManager(
        survival_slots=3
    )

    # 每个元素表示对应时隙到达的请求数量。
    request_trace = [
        2,
        0,
        0,
        1,
        0,
        0,
        0,
        3,
        0,
        1,
    ]

    total_cold_starts = 0
    total_warm_hits = 0
    total_delay_ms = 0.0

    print("=" * 100)
    print("Serverless 容器冷启动与生存窗口演示")
    print("=" * 100)

    print(
        "时隙  请求数  处理前状态      处理后状态      "
        "空闲计数  冷启动  温命中  销毁  总时延(ms)"
    )
    print("-" * 100)

    for time_slot, request_count in enumerate(request_trace):
        result = manager.process_slot(
            instance=instance,
            function=function,
            request_count=request_count,
            keep_warm=False,
        )

        total_cold_starts += int(
            result.cold_start_occurred
        )

        total_warm_hits += int(
            result.warm_hit
        )

        total_delay_ms += result.total_delay_ms

        print(
            f"{time_slot:4d}"
            f"  {request_count:6d}"
            f"  {result.before_status.value:14s}"
            f"  {result.after_status.value:14s}"
            f"  {result.idle_slots_after:8d}"
            f"  {str(result.cold_start_occurred):6s}"
            f"  {str(result.warm_hit):6s}"
            f"  {str(result.expired):4s}"
            f"  {result.total_delay_ms:10.2f}"
        )

    print("-" * 100)
    print(f"冷启动总次数：{total_cold_starts}")
    print(f"温实例命中次数：{total_warm_hits}")
    print(f"总时延：{total_delay_ms:.2f} ms")
    print("=" * 100)


if __name__ == "__main__":
    main()