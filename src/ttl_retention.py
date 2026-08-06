"""
ttl_retention.py

轨道边缘 Serverless SFC 的离散 TTL（Time-To-Live）
容器保留状态管理器。

本模块只负责“跨快时隙、跨慢窗口”的温备用生命周期状态，
不负责请求路由、故障判断或奖励计算。

核心语义
--------
1. 一个动作选择一个明确的 TTL 快时隙数；
2. TTL=0 表示立即释放全部温备用容器；
3. TTL>0 表示在目标备用节点上预热/刷新容器；
4. 每经过一个快时隙，剩余 TTL 减少 1；
5. 剩余 TTL 变为 0 时，容器到期释放；
6. 主实例不由本模块管理，本模块只管理备用温实例。

为什么不直接使用 0/1/2/3 个慢窗口
-----------------------------------
如果智能体每个慢窗口都重新决策并刷新 TTL，
“1 个慢窗口”和“始终温热”在多数时间内会变得几乎相同。
因此使用可配置的快时隙选项，例如：

    [0, 3, 6, 10]

当慢周期为 10 个快时隙时，四个动作分别表示：

    0：不保留；
    1：短期保留 3 个快时隙；
    2：中期保留 6 个快时隙；
    3：完整保留 10 个快时隙。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Mapping


# (function_id, node_id)
ReplicaKey = tuple[int, int]


class TTLAction(IntEnum):
    """
    Double-DQN 的四个离散 TTL 动作。

    动作编号只表示在 ttl_slot_options 中的索引，
    实际 TTL 快时隙数由配置决定。
    """

    TTL_0 = 0
    TTL_1 = 1
    TTL_2 = 2
    TTL_3 = 3


@dataclass(frozen=True)
class TTLUpdateResult:
    """
    一次 TTL 动作应用后的状态变化。
    """

    action: TTLAction
    ttl_slots: int

    target_keys: tuple[ReplicaKey, ...]
    newly_prewarmed_keys: tuple[ReplicaKey, ...]
    refreshed_keys: tuple[ReplicaKey, ...]
    released_keys: tuple[ReplicaKey, ...]


@dataclass(frozen=True)
class TTLSnapshot:
    """
    当前全部温备用容器及其剩余 TTL。
    """

    remaining_slots: tuple[
        tuple[ReplicaKey, int],
        ...
    ]

    @property
    def warm_keys(self) -> tuple[ReplicaKey, ...]:
        return tuple(
            key
            for key, _
            in self.remaining_slots
        )


def _normalize_replica_keys(
    keys: Iterable[ReplicaKey],
) -> tuple[ReplicaKey, ...]:
    """
    校验、去重并排序函数副本键。
    """

    normalized: set[ReplicaKey] = set()

    for raw_key in keys:
        if len(raw_key) != 2:
            raise ValueError(
                "副本键必须是(function_id, node_id)。"
            )

        function_id = int(raw_key[0])
        node_id = int(raw_key[1])

        if function_id < 0:
            raise ValueError(
                "function_id不能小于0。"
            )

        if node_id < 0:
            raise ValueError(
                "node_id不能小于0。"
            )

        normalized.add(
            (function_id, node_id)
        )

    return tuple(sorted(normalized))


class TTLRetentionManager:
    """
    跨时隙温备用容器 TTL 管理器。
    """

    def __init__(
        self,
        ttl_slot_options: Iterable[int],
    ) -> None:
        """
        Parameters
        ----------
        ttl_slot_options:
            四个动作对应的 TTL 快时隙数。

            例如：
                [0, 3, 6, 10]

            必须满足：
            1. 恰好包含4个值；
            2. 第一个值为0；
            3. 严格递增；
            4. 所有值均为非负整数。
        """

        options = tuple(
            int(value)
            for value in ttl_slot_options
        )

        if len(options) != len(TTLAction):
            raise ValueError(
                "TTL选项数量必须等于4。"
            )

        if options[0] != 0:
            raise ValueError(
                "第一个TTL选项必须为0。"
            )

        if any(value < 0 for value in options):
            raise ValueError(
                "TTL快时隙数不能小于0。"
            )

        if any(
            left >= right
            for left, right
            in zip(
                options[:-1],
                options[1:],
                strict=True,
            )
        ):
            raise ValueError(
                "TTL选项必须严格递增。"
            )

        self.ttl_slot_options = options

        self._remaining_slots: dict[
            ReplicaKey,
            int,
        ] = {}

    def reset(self) -> None:
        """
        清空一个 Episode 中的全部温备用状态。
        """

        self._remaining_slots.clear()

    def ttl_slots_for_action(
        self,
        action: int | TTLAction,
    ) -> int:
        """
        返回动作对应的 TTL 快时隙数。
        """

        try:
            selected_action = TTLAction(
                int(action)
            )
        except ValueError as error:
            raise ValueError(
                "TTL动作必须位于0～3。"
            ) from error

        return self.ttl_slot_options[
            int(selected_action)
        ]

    def is_warm(
        self,
        key: ReplicaKey,
    ) -> bool:
        """
        判断某个备用函数实例当前是否温热。
        """

        normalized_key = (
            _normalize_replica_keys([key])[0]
        )

        return normalized_key in (
            self._remaining_slots
        )

    def remaining_slots_for(
        self,
        key: ReplicaKey,
    ) -> int:
        """
        返回某个备用实例的剩余 TTL。
        """

        normalized_key = (
            _normalize_replica_keys([key])[0]
        )

        return int(
            self._remaining_slots.get(
                normalized_key,
                0,
            )
        )

    def apply_action(
        self,
        action: int | TTLAction,
        target_backup_keys: Iterable[
            ReplicaKey
        ],
    ) -> TTLUpdateResult:
        """
        在慢时间尺度决策点应用一个 TTL 动作。

        TTL=0：
            立即释放全部温备用容器。

        TTL>0：
            对当前目标备用容器执行预热或 TTL 刷新。
            与当前目标无关的旧容器继续按照原 TTL 倒计时。
        """

        try:
            selected_action = TTLAction(
                int(action)
            )
        except ValueError as error:
            raise ValueError(
                "TTL动作必须位于0～3。"
            ) from error

        target_keys = _normalize_replica_keys(
            target_backup_keys
        )

        ttl_slots = self.ttl_slots_for_action(
            selected_action
        )

        if ttl_slots == 0:
            released_keys = tuple(
                sorted(
                    self._remaining_slots
                )
            )

            self._remaining_slots.clear()

            return TTLUpdateResult(
                action=selected_action,
                ttl_slots=0,
                target_keys=target_keys,
                newly_prewarmed_keys=(),
                refreshed_keys=(),
                released_keys=released_keys,
            )

        newly_prewarmed: list[
            ReplicaKey
        ] = []

        refreshed: list[
            ReplicaKey
        ] = []

        for key in target_keys:
            if key in self._remaining_slots:
                refreshed.append(key)
            else:
                newly_prewarmed.append(key)

            # 这里使用“重新设为所选TTL”，
            # 因而智能体可以主动缩短或延长保留时间。
            self._remaining_slots[key] = (
                ttl_slots
            )

        return TTLUpdateResult(
            action=selected_action,
            ttl_slots=ttl_slots,
            target_keys=target_keys,
            newly_prewarmed_keys=tuple(
                newly_prewarmed
            ),
            refreshed_keys=tuple(
                refreshed
            ),
            released_keys=(),
        )

    def retain_after_request(
        self,
        executed_backup_keys: Iterable[
            ReplicaKey
        ],
        action: int | TTLAction,
    ) -> tuple[ReplicaKey, ...]:
        """
        请求在备用节点执行后，按照当前动作刷新生存期。

        返回本次请求结束后新进入温状态的备用实例。

        TTL=0：
            请求处理完成后不继续保留。

        TTL>0：
            将实际执行过的备用实例 TTL 刷新为所选值。
        """

        keys = _normalize_replica_keys(
            executed_backup_keys
        )

        ttl_slots = self.ttl_slots_for_action(
            action
        )

        if ttl_slots == 0:
            for key in keys:
                self._remaining_slots.pop(
                    key,
                    None,
                )

            return ()

        newly_retained: list[
            ReplicaKey
        ] = []

        for key in keys:
            if key not in self._remaining_slots:
                newly_retained.append(key)

            self._remaining_slots[key] = (
                ttl_slots
            )

        return tuple(newly_retained)

    def tick(
        self,
        elapsed_slots: int = 1,
    ) -> tuple[ReplicaKey, ...]:
        """
        推进若干个快时隙并释放到期容器。

        应在完成当前快时隙的内存统计之后调用。
        """

        if elapsed_slots <= 0:
            raise ValueError(
                "elapsed_slots必须大于0。"
            )

        expired: list[
            ReplicaKey
        ] = []

        for key in tuple(
            self._remaining_slots
        ):
            remaining = (
                self._remaining_slots[key]
                - elapsed_slots
            )

            if remaining <= 0:
                expired.append(key)
                del self._remaining_slots[key]
            else:
                self._remaining_slots[key] = (
                    remaining
                )

        return tuple(sorted(expired))

    def snapshot(self) -> TTLSnapshot:
        """
        返回不可变状态快照。
        """

        return TTLSnapshot(
            remaining_slots=tuple(
                sorted(
                    (
                        key,
                        int(remaining),
                    )
                    for key, remaining
                    in self._remaining_slots.items()
                )
            )
        )

    def active_standby_memory_mb(
        self,
        function_memory_mb: Mapping[
            int,
            float,
        ],
    ) -> float:
        """
        计算当前温备用容器占用的内存。
        """

        total = 0.0

        for function_id, _ in (
            self._remaining_slots
        ):
            if function_id not in (
                function_memory_mb
            ):
                raise KeyError(
                    f"缺少function_id="
                    f"{function_id}的内存配置。"
                )

            memory_mb = float(
                function_memory_mb[
                    function_id
                ]
            )

            if memory_mb < 0:
                raise ValueError(
                    "函数内存不能小于0。"
                )

            total += memory_mb

        return total
