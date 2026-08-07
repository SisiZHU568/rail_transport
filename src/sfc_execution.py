"""
sfc_execution.py

本文件负责模拟一个完整 SFC 请求的顺序执行。

一个 SFC 请求的执行过程包括：

1. 将输入数据传输到第一个函数所在节点；
2. 执行第一个函数；
3. 根据 output_ratio 更新输出数据量；
4. 将输出数据传输到第二个函数所在节点；
5. 依次执行剩余函数；
6. 将最终结果返回原始接入 MEC。

当前阶段模拟一个请求的执行，
后续会扩展到批量请求和并发请求。
"""

from dataclasses import dataclass,replace

from src.entities import ServerlessFunction, SFCType
from src.network import TransferNetworkProtocol


@dataclass(frozen=True)
class FunctionExecutionRecord:
    """
    SFC 中一个函数的执行记录。
    """

    # 函数在 SFC 中的顺序，从0开始
    order_index: int

    # 函数编号
    function_id: int

    # 函数名称
    function_name: str

    # 函数部署在哪个 MEC
    node_id: int

    # 函数执行前，数据原本位于哪个 MEC
    previous_node_id: int

    # 函数接收到的输入数据量
    input_size_mb: float

    # 将输入数据传输到本函数所在节点的时延
    transmission_delay_ms: float

    # 本函数的冷启动时延
    cold_start_delay_ms: float

    # 本函数的温执行时延
    execution_delay_ms: float

    # 函数执行完成后的输出数据量
    output_size_mb: float

    @property
    def total_function_stage_delay_ms(self) -> float:
        """
        返回该函数阶段的总时延。

        包括：

        1. 输入数据传输；
        2. 冷启动；
        3. 函数执行。
        """

        return (
            self.transmission_delay_ms
            + self.cold_start_delay_ms
            + self.execution_delay_ms
        )


@dataclass(frozen=True)
class SFCExecutionResult:
    """
    一个完整 SFC 请求的执行结果。
    """

    sfc_id: int
    sfc_name: str

    # 原始数据所在 MEC
    source_node_id: int

    # 每个函数的部署节点
    placement_node_ids: tuple[int, ...]

    # 每个函数的详细执行记录
    function_records: tuple[FunctionExecutionRecord, ...]

    # 最终结果数据量
    final_output_size_mb: float

    # 最终结果返回源 MEC 的时延
    return_transmission_delay_ms: float

    # 全部函数间传输和返回传输的总时延
    total_transmission_delay_ms: float

    # 输入、函数间和结果返回产生的总数据传输成本
    total_routing_cost: float

    # 所有函数的冷启动总时延
    total_cold_start_delay_ms: float

    # 所有函数的执行总时延
    total_execution_delay_ms: float

    # 完整端到端时延
    total_end_to_end_delay_ms: float

    # SFC 时延要求
    deadline_ms: float

    # 是否满足时延要求
    deadline_met: bool


def execute_sfc_request(
    functions: list[ServerlessFunction],
    sfc: SFCType,
    placement_node_ids: list[int],
    source_node_id: int,
    input_size_mb: float,
    network: TransferNetworkProtocol,
    cold_start_function_ids: set[int] | None = None,
    return_result_to_source: bool = True,
) -> SFCExecutionResult:
    """
    执行一个完整的 SFC 请求。

    Parameters
    ----------
    functions:
        系统中所有可用的 Serverless 函数。

    sfc:
        当前需要执行的 SFC。

    placement_node_ids:
        每个 SFC 函数的部署节点。

        例如：

            sfc.function_ids = [0, 1, 2]
            placement_node_ids = [0, 1, 2]

        表示：

            函数0部署在 MEC-1；
            函数1部署在 MEC-2；
            函数2部署在 MEC-3。

    source_node_id:
        原始输入数据所在 MEC。

    input_size_mb:
        SFC 原始输入数据量。

    network:
        实现统一传输接口的轨旁或边缘—云网络模型。

    cold_start_function_ids:
        当前请求执行时需要冷启动的函数编号集合。

        例如：

            {0, 2}

        表示函数0和函数2会发生冷启动，
        函数1已经存在温实例。

        None 表示所有函数都已经处于温状态。

    return_result_to_source:
        SFC 完成后，是否将最终结果返回源 MEC。

    Returns
    -------
    SFCExecutionResult:
        完整的函数执行和端到端时延结果。
    """

    if input_size_mb < 0:
        raise ValueError("SFC 输入数据量不能小于 0。")

    if len(placement_node_ids) != len(sfc.function_ids):
        raise ValueError(
            "函数部署节点数量必须与 SFC 中的函数数量一致。"
        )

    # 如果调用者没有提供冷启动集合，
    # 则使用空集合，表示所有函数都为温实例。
    if cold_start_function_ids is None:
        cold_start_function_ids = set()

    # 建立函数编号到函数对象的映射。
    function_map: dict[int, ServerlessFunction] = {}

    for function in functions:
        if function.function_id in function_map:
            raise ValueError(
                f"函数编号 {function.function_id} 重复。"
            )

        function_map[function.function_id] = function

    # 检查 SFC 中所有函数是否都存在。
    for function_id in sfc.function_ids:
        if function_id not in function_map:
            raise KeyError(
                f"找不到 function_id={function_id} 的函数。"
            )

    # 通过零数据同节点传输检查源节点是否存在。
    # 公共网络接口不要求中心云具有轨道跳数。
    network.transfer_delay_ms(
        data_size_mb=0.0,
        source_node_id=source_node_id,
        destination_node_id=source_node_id,
    )

    records: list[FunctionExecutionRecord] = []

    current_node_id = source_node_id
    current_data_size_mb = input_size_mb

    total_transmission_delay_ms = 0.0
    total_routing_cost = 0.0
    total_cold_start_delay_ms = 0.0
    total_execution_delay_ms = 0.0

    # 按照 SFC 中定义的顺序依次执行函数。
    for order_index, function_id in enumerate(
        sfc.function_ids
    ):
        function = function_map[function_id]

        destination_node_id = (
            placement_node_ids[order_index]
        )

        # 将上一个阶段产生的数据传输到当前函数节点。
        transmission_delay_ms = network.transfer_delay_ms(
            data_size_mb=current_data_size_mb,
            source_node_id=current_node_id,
            destination_node_id=destination_node_id,
        )
        routing_cost = network.transfer_cost(
            data_size_mb=current_data_size_mb,
            source_node_id=current_node_id,
            destination_node_id=destination_node_id,
        )

        # 判断当前函数是否需要冷启动。
        if function_id in cold_start_function_ids:
            cold_start_delay_ms = (
                function.cold_start_time_ms
            )
        else:
            cold_start_delay_ms = 0.0

        # 当前模拟一个请求，
        # 所以执行时延等于单请求温执行时间。
        execution_delay_ms = (
            function.warm_exec_time_ms
        )

        # 函数执行后，数据量按照 output_ratio 变化。
        output_size_mb = (
            current_data_size_mb
            * function.output_ratio
        )

        record = FunctionExecutionRecord(
            order_index=order_index,
            function_id=function.function_id,
            function_name=function.name,
            node_id=destination_node_id,
            previous_node_id=current_node_id,
            input_size_mb=current_data_size_mb,
            transmission_delay_ms=(
                transmission_delay_ms
            ),
            cold_start_delay_ms=cold_start_delay_ms,
            execution_delay_ms=execution_delay_ms,
            output_size_mb=output_size_mb,
        )

        records.append(record)

        total_transmission_delay_ms += (
            transmission_delay_ms
        )
        total_routing_cost += routing_cost

        total_cold_start_delay_ms += (
            cold_start_delay_ms
        )

        total_execution_delay_ms += (
            execution_delay_ms
        )

        # 当前函数执行结束后，
        # 数据位于当前函数所在 MEC。
        current_node_id = destination_node_id

        # 下一函数接收当前函数的输出数据。
        current_data_size_mb = output_size_mb

    # SFC 全部函数执行完成后，
    # 根据配置决定是否将最终结果返回源 MEC。
    if return_result_to_source:
        return_transmission_delay_ms = (
            network.transfer_delay_ms(
                data_size_mb=current_data_size_mb,
                source_node_id=current_node_id,
                destination_node_id=source_node_id,
            )
        )
        return_routing_cost = network.transfer_cost(
            data_size_mb=current_data_size_mb,
            source_node_id=current_node_id,
            destination_node_id=source_node_id,
        )
    else:
        return_transmission_delay_ms = 0.0
        return_routing_cost = 0.0

    total_transmission_delay_ms += (
        return_transmission_delay_ms
    )
    total_routing_cost += return_routing_cost

    total_end_to_end_delay_ms = (
        total_transmission_delay_ms
        + total_cold_start_delay_ms
        + total_execution_delay_ms
    )

    deadline_met = (
        total_end_to_end_delay_ms
        <= sfc.deadline_ms
    )

    return SFCExecutionResult(
        sfc_id=sfc.sfc_id,
        sfc_name=sfc.name,
        source_node_id=source_node_id,
        placement_node_ids=tuple(
            placement_node_ids
        ),
        function_records=tuple(records),
        final_output_size_mb=current_data_size_mb,
        return_transmission_delay_ms=(
            return_transmission_delay_ms
        ),
        total_transmission_delay_ms=(
            total_transmission_delay_ms
        ),
        total_routing_cost=total_routing_cost,
        total_cold_start_delay_ms=(
            total_cold_start_delay_ms
        ),
        total_execution_delay_ms=(
            total_execution_delay_ms
        ),
        total_end_to_end_delay_ms=(
            total_end_to_end_delay_ms
        ),
        deadline_ms=sfc.deadline_ms,
        deadline_met=deadline_met,
    )

def execute_sfc_batch(
    functions: list[ServerlessFunction],
    sfc: SFCType,
    placement_node_ids: list[int],
    source_node_id: int,
    input_size_mb_per_request: float,
    request_count: int,
    network: TransferNetworkProtocol,
    cold_start_function_ids: set[int] | None = None,
    return_result_to_source: bool = True,
) -> SFCExecutionResult:
    """
    执行一批同类型 SFC 请求。

    当前采用简单的串行批量处理模型：

        批量输入数据量
        =
        单请求输入数据量 × 请求数量

        批量函数执行时间
        =
        单请求温执行时间 × 请求数量

    冷启动只发生一次，不会随着请求数量重复计算。

    例如：

        3个请求同时到达；
        某函数冷启动时间为300 ms；
        单请求执行时间为15 ms。

    则该函数阶段：

        冷启动时间 = 300 ms
        执行时间 = 15 × 3 = 45 ms

    Parameters
    ----------
    functions:
        所有 Serverless 函数。

    sfc:
        当前执行的服务功能链。

    placement_node_ids:
        每个函数的部署节点。

    source_node_id:
        原始数据所在 MEC。

    input_size_mb_per_request:
        单个请求的输入数据量。

    request_count:
        当前批次的请求数量。

    network:
        MEC 网络模型。

    cold_start_function_ids:
        当前需要发生冷启动的函数编号。

    return_result_to_source:
        是否将最终结果返回源 MEC。

    Returns
    -------
    SFCExecutionResult:
        批量请求的完整执行结果。
    """

    if input_size_mb_per_request < 0:
        raise ValueError(
            "单请求输入数据量不能小于 0。"
        )

    if request_count <= 0:
        raise ValueError(
            "批量执行时 request_count 必须大于 0。"
        )

    # 为本次批量执行创建临时函数参数。
    #
    # 只放大温执行时间，
    # 不改变冷启动时间、内存和镜像大小。
    batch_functions: list[ServerlessFunction] = []

    for function in functions:
        batch_function = replace(
            function,
            warm_exec_time_ms=(
                function.warm_exec_time_ms
                * request_count
            ),
        )

        batch_functions.append(batch_function)

    # 批量输入数据量。
    batch_input_size_mb = (
        input_size_mb_per_request
        * request_count
    )

    return execute_sfc_request(
        functions=batch_functions,
        sfc=sfc,
        placement_node_ids=placement_node_ids,
        source_node_id=source_node_id,
        input_size_mb=batch_input_size_mb,
        network=network,
        cold_start_function_ids=cold_start_function_ids,
        return_result_to_source=return_result_to_source,
    )
