"""
cold_start.py

本文件负责模拟 Serverless 函数容器的生命周期。

主要状态变化如下：

    ABSENT
      ↓ 请求到达或主动预热
    COLD_STARTING
      ↓ 冷启动完成
    WARM
      ↓ 开始处理请求
    BUSY
      ↓ 请求处理完成
    WARM
      ↓ 连续空闲达到生存窗口
    ABSENT

当前模型采用“时隙级仿真”。

为了降低第一版代码复杂度，冷启动过程在一个时隙内部完成，
但冷启动产生的延迟仍会被完整记录。
后续可以扩展成跨多个时隙的启动过程。
"""

from dataclasses import dataclass

from src.entities import (
    FunctionInstance,
    InstanceStatus,
    ServerlessFunction,
)


@dataclass(frozen=True)
class LifecycleResult:
    """
    一个时隙内的容器生命周期处理结果。

    frozen=True 表示结果对象创建后不可修改，
    可以避免后续代码无意间改变已经产生的仿真结果。
    """

    # 本时隙到达的请求数量
    request_count: int

    # 处理本时隙之前的实例状态
    before_status: InstanceStatus

    # 处理本时隙之后的实例状态
    after_status: InstanceStatus

    # 是否发生了冷启动
    cold_start_occurred: bool

    # 是否命中了已有的温实例
    warm_hit: bool

    # 是否属于主动预热，而不是请求触发
    prewarm_occurred: bool

    # 本时隙结束时容器是否被销毁
    expired: bool

    # 冷启动产生的额外时延，单位为毫秒
    cold_start_delay_ms: float

    # 请求执行产生的总时延，单位为毫秒
    execution_delay_ms: float

    # 本时隙结束后的连续空闲时隙数量
    idle_slots_after: int

    @property
    def total_delay_ms(self) -> float:
        """
        返回冷启动时延与执行时延之和。
        """

        return (
            self.cold_start_delay_ms
            + self.execution_delay_ms
        )


class ContainerLifecycleManager:
    """
    Serverless 容器生命周期管理器。

    它根据以下信息更新函数实例状态：

    1. 当前是否有请求；
    2. 是否要求主动保温；
    3. 容器已经空闲了多久；
    4. 容器生存窗口是多少。
    """

    def __init__(self, survival_slots: int) -> None:
        """
        Parameters
        ----------
        survival_slots:
            容器允许连续空闲的最大时隙数量。

            例如 survival_slots=3：

            第1个空闲时隙：容器仍然为 WARM；
            第2个空闲时隙：容器仍然为 WARM；
            第3个空闲时隙：容器被销毁，变为 ABSENT。
        """

        if survival_slots <= 0:
            raise ValueError("容器生存窗口必须大于 0。")

        self.survival_slots = survival_slots

    def process_slot(
        self,
        instance: FunctionInstance,
        function: ServerlessFunction,
        request_count: int,
        keep_warm: bool = False,
    ) -> LifecycleResult:
        """
        处理一个快时隙中的容器状态变化。

        Parameters
        ----------
        instance:
            当前节点上的函数实例。

        function:
            函数本身的资源与时延参数。

        request_count:
            当前时隙到达的请求数量。

        keep_warm:
            是否主动保温。

            False:
                没有请求时正常累计空闲时间。

            True:
                即使没有真实请求，也通过预热或保活维持温实例。

        Returns
        -------
        LifecycleResult:
            当前时隙的处理结果。
        """

        if request_count < 0:
            raise ValueError("请求数量不能小于 0。")

        # 记录进入本时隙时的容器状态。
        before_status = instance.status

        # 初始化本时隙的结果变量。
        cold_start_occurred = False
        warm_hit = False
        prewarm_occurred = False
        expired = False
        cold_start_delay_ms = 0.0
        execution_delay_ms = 0.0

        # 发生故障的实例不能处理请求，也不能执行预热。
        if instance.status == InstanceStatus.FAILED:
            if request_count > 0 or keep_warm:
                raise RuntimeError(
                    "FAILED 状态的函数实例不能处理请求或执行预热。"
                )

            return LifecycleResult(
                request_count=request_count,
                before_status=before_status,
                after_status=instance.status,
                cold_start_occurred=False,
                warm_hit=False,
                prewarm_occurred=False,
                expired=False,
                cold_start_delay_ms=0.0,
                execution_delay_ms=0.0,
                idle_slots_after=instance.idle_slots,
            )

        # 情况一：当前时隙有真实请求。
        if request_count > 0:
            # 当前不存在容器，需要先冷启动。
            if instance.status == InstanceStatus.ABSENT:
                instance.status = InstanceStatus.COLD_STARTING
                instance.remaining_startup_time_ms = (
                    function.cold_start_time_ms
                )

                cold_start_occurred = True
                cold_start_delay_ms = function.cold_start_time_ms

            # 当前已经有可复用的温实例。
            elif instance.status in {
                InstanceStatus.WARM,
                InstanceStatus.BUSY,
            }:
                warm_hit = True

            # 容器进入忙碌状态。
            instance.status = InstanceStatus.BUSY

            # 当前采用简单的串行批量执行模型：
            #
            # 总执行时间
            # =
            # 单请求温执行时间 × 请求数量
            execution_delay_ms = (
                function.warm_exec_time_ms
                * request_count
            )

            # 当前时隙内处理完成，重新回到温状态。
            instance.status = InstanceStatus.WARM
            instance.idle_slots = 0
            instance.remaining_startup_time_ms = 0.0

        # 情况二：当前时隙没有真实请求。
        else:
            # 要求实例保持温热状态。
            if keep_warm:
                # 容器不存在时，需要主动预热。
                if instance.status == InstanceStatus.ABSENT:
                    instance.status = InstanceStatus.COLD_STARTING

                    cold_start_occurred = True
                    prewarm_occurred = True
                    cold_start_delay_ms = (
                        function.cold_start_time_ms
                    )

                    # 当前简化模型中，预热在本时隙内完成。
                    instance.status = InstanceStatus.WARM

                # 如果原本已经是温实例，只需继续保温。
                elif instance.status in {
                    InstanceStatus.WARM,
                    InstanceStatus.BUSY,
                    InstanceStatus.COLD_STARTING,
                }:
                    instance.status = InstanceStatus.WARM

                # 保活后，连续空闲计数清零。
                instance.idle_slots = 0
                instance.remaining_startup_time_ms = 0.0

            # 没有请求，也没有主动保温。
            else:
                if instance.status in {
                    InstanceStatus.WARM,
                    InstanceStatus.BUSY,
                }:
                    # 空闲时隙数量增加。
                    instance.status = InstanceStatus.WARM
                    instance.idle_slots += 1

                    # 达到生存窗口后，容器被回收。
                    if instance.idle_slots >= self.survival_slots:
                        instance.status = InstanceStatus.ABSENT
                        instance.idle_slots = 0
                        instance.remaining_startup_time_ms = 0.0
                        expired = True

        return LifecycleResult(
            request_count=request_count,
            before_status=before_status,
            after_status=instance.status,
            cold_start_occurred=cold_start_occurred,
            warm_hit=warm_hit,
            prewarm_occurred=prewarm_occurred,
            expired=expired,
            cold_start_delay_ms=cold_start_delay_ms,
            execution_delay_ms=execution_delay_ms,
            idle_slots_after=instance.idle_slots,
        )