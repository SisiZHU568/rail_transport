"""
retention_policies.py

本文件定义三种基础容器保留策略：

1. OnDemandPolicy：
   按需启动。只要出现一个空闲时隙，就销毁容器。

2. FixedWindowPolicy：
   固定生存窗口。连续空闲若干个时隙后销毁容器。

3. AlwaysWarmPolicy：
   始终保温。没有真实请求时，也主动维持温容器。

后续强化学习算法会与这些基础策略进行对比。
"""

from abc import ABC, abstractmethod

from src.entities import FunctionInstance


class RetentionPolicy(ABC):
    """
    容器保留策略的抽象基类。

    抽象基类的作用是规定所有保留策略必须具有：

    1. 策略编号；
    2. 策略显示名称；
    3. 生存窗口；
    4. 是否主动保温的决策方法。

    后续增加新的策略时，应继承本类。
    """

    def __init__(
        self,
        policy_id: str,
        display_name: str,
        survival_slots: int,
    ) -> None:
        """
        Parameters
        ----------
        policy_id:
            策略的英文编号，主要用于 CSV 和绘图。

        display_name:
            策略的中文名称，主要用于终端显示。

        survival_slots:
            容器允许连续空闲的最大时隙数。
        """

        if not policy_id:
            raise ValueError("policy_id 不能为空。")

        if not display_name:
            raise ValueError("display_name 不能为空。")

        if survival_slots <= 0:
            raise ValueError("survival_slots 必须大于 0。")

        self.policy_id = policy_id
        self.display_name = display_name
        self.survival_slots = survival_slots

    @abstractmethod
    def should_keep_warm(
        self,
        time_slot: int,
        request_count: int,
        instance: FunctionInstance,
    ) -> bool:
        """
        决定当前时隙是否需要主动保温。

        Returns
        -------
        bool:
            True 表示主动预热或保温；
            False 表示不主动干预，由生存窗口决定何时销毁。
        """

        raise NotImplementedError


class OnDemandPolicy(RetentionPolicy):
    """
    按需启动策略。

    工作方式：

    - 有请求时启动容器；
    - 没有请求时不主动保温；
    - 经过一个空闲时隙后立即销毁。

    该策略保留资源成本最低，但容易产生大量冷启动。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="on_demand",
            display_name="按需启动",
            survival_slots=1,
        )

    def should_keep_warm(
        self,
        time_slot: int,
        request_count: int,
        instance: FunctionInstance,
    ) -> bool:
        """
        按需策略永远不会主动保温。
        """

        return False


class FixedWindowPolicy(RetentionPolicy):
    """
    固定生存窗口策略。

    请求结束后，容器继续保持温热状态若干个时隙。
    如果生存窗口内再次有请求，就可以复用温实例。
    """

    def __init__(self, window_slots: int) -> None:
        super().__init__(
            policy_id="fixed_window",
            display_name=f"固定窗口({window_slots})",
            survival_slots=window_slots,
        )

    def should_keep_warm(
        self,
        time_slot: int,
        request_count: int,
        instance: FunctionInstance,
    ) -> bool:
        """
        固定窗口策略不主动发送保活请求。

        容器是否继续保留，由 ContainerLifecycleManager
        中的 survival_slots 自动决定。
        """

        return False


class AlwaysWarmPolicy(RetentionPolicy):
    """
    始终保温策略。

    即使没有真实请求，也会主动预热或发送保活请求，
    保证函数容器始终处于 WARM 状态。

    该策略能够减少用户冷启动时延，但资源占用较大。
    """

    def __init__(self) -> None:
        super().__init__(
            policy_id="always_warm",
            display_name="始终保温",

            # 实际上始终保温策略不会依赖生存窗口。
            # 这里仍然提供一个合法值，以满足生命周期管理器要求。
            survival_slots=1,
        )

    def should_keep_warm(
        self,
        time_slot: int,
        request_count: int,
        instance: FunctionInstance,
    ) -> bool:
        """
        始终返回 True，表示每个时隙都维持温实例。
        """

        return True