"""
workload.py

本文件负责描述每个快时隙到达多少个 SFC 请求。

当前首先实现确定性请求轨迹：

    [0, 1, 0, 2, 1, ...]

其中：

    0 表示当前时隙没有请求；
    1 表示当前时隙有 1 个请求；
    2 表示当前时隙有 2 个请求。

确定性请求轨迹便于调试和复现实验。
后续会增加泊松请求、突发请求和真实数据集请求轨迹。
"""


class DeterministicWorkload:
    """
    确定性请求负载模型。

    Parameters
    ----------
    request_trace:
        每个时隙的请求数量。

    repeat:
        请求轨迹结束后是否从头循环。

        True:
            轨迹循环使用。

        False:
            轨迹结束后，后续时隙请求数量都为 0。
    """

    def __init__(
        self,
        request_trace: list[int],
        repeat: bool = False,
    ) -> None:
        """
        创建确定性请求负载。
        """

        if len(request_trace) == 0:
            raise ValueError("请求轨迹不能为空。")

        if any(
            not isinstance(request_count, int)
            for request_count in request_trace
        ):
            raise TypeError("请求数量必须是整数。")

        if any(
            request_count < 0
            for request_count in request_trace
        ):
            raise ValueError("请求数量不能小于 0。")

        # 转换为元组，防止外部代码修改原始列表后，
        # 影响已经创建好的负载模型。
        self.request_trace = tuple(request_trace)

        self.repeat = repeat

    @property
    def trace_length(self) -> int:
        """
        返回基础请求轨迹的长度。
        """

        return len(self.request_trace)

    def request_count(self, time_slot: int) -> int:
        """
        查询指定时隙的请求数量。

        Parameters
        ----------
        time_slot:
            快时隙编号。

        Returns
        -------
        int:
            当前时隙到达的请求数量。
        """

        if time_slot < 0:
            raise ValueError("time_slot 不能小于 0。")

        # 循环请求轨迹。
        if self.repeat:
            trace_index = (
                time_slot % len(self.request_trace)
            )

            return self.request_trace[trace_index]

        # 不循环，并且时隙已经超过轨迹长度。
        if time_slot >= len(self.request_trace):
            return 0

        return self.request_trace[time_slot]