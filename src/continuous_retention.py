"""管理 DPPO 连续动作给出的主副本与备用副本保留时间。"""

from collections.abc import Collection
import math


# 字典键统一使用“VNF 编号、计算节点编号”，一个 VNF 可以在多个节点保持温热。
ReplicaKey = tuple[int, int]


def _validate_nonnegative_integer(value: object, name: str) -> int:
    """校验时隙和编号，并显式排除会被 Python 当作整数的布尔值。"""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _validate_retention_seconds(value: object) -> float:
    """把动作值转换为可计算的非负有限秒数。"""

    if isinstance(value, bool):
        raise ValueError("retention seconds must be finite and non-negative.")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "retention seconds must be finite and non-negative."
        ) from error
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("retention seconds must be finite and non-negative.")
    return seconds


class ContinuousRetentionTracker:
    """跟踪每个 VNF 副本在快时间尺度上的绝对到期时隙。

    慢层在 ``slot`` 完成决策后，连续保留从下一个完整快时隙开始计算。
    例如在时隙 2 选择保留 1 个时隙，则时隙 3 仍温热，时隙 4 到期。
    这个约定与 Task 5 的研究示例一致，也避免把决策所在时隙重复计费。
    """

    def __init__(self, slot_seconds: float) -> None:
        """保存快时隙长度；秒数稍后会据此向上取整为完整时隙。"""

        if isinstance(slot_seconds, bool):
            raise ValueError("slot_seconds must be a positive finite number.")
        try:
            normalized_slot_seconds = float(slot_seconds)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "slot_seconds must be a positive finite number."
            ) from error
        if (
            not math.isfinite(normalized_slot_seconds)
            or normalized_slot_seconds <= 0.0
        ):
            raise ValueError("slot_seconds must be a positive finite number.")

        self.slot_seconds = normalized_slot_seconds
        self._expiry_slots: dict[ReplicaKey, int] = {}

    def apply_intent(
        self,
        *,
        slot: int,
        function_id: int,
        primary_node_id: int,
        backup_node_ids: tuple[int, ...],
        primary_seconds: float,
        backup_seconds: float,
    ) -> None:
        """应用一次逐 VNF 意图，并分别刷新主副本和所有备用副本。

        刷新使用较晚的到期时隙，因此连续决策可以延长保留，但不会因为
        新动作更短而提前释放当前仍有效的温热实例。
        """

        normalized_slot = _validate_nonnegative_integer(slot, "slot")
        normalized_function_id = _validate_nonnegative_integer(
            function_id,
            "function_id",
        )
        normalized_primary_id = _validate_nonnegative_integer(
            primary_node_id,
            "primary_node_id",
        )
        normalized_backup_ids = tuple(
            _validate_nonnegative_integer(node_id, "backup node ID")
            for node_id in backup_node_ids
        )
        if len(normalized_backup_ids) not in (1, 2):
            raise ValueError("backup_node_ids must contain one or two nodes.")

        all_node_ids = (normalized_primary_id,) + normalized_backup_ids
        if len(set(all_node_ids)) != len(all_node_ids):
            raise ValueError("primary and backup node IDs must be unique.")

        normalized_primary_seconds = _validate_retention_seconds(
            primary_seconds
        )
        normalized_backup_seconds = _validate_retention_seconds(
            backup_seconds
        )
        self._purge_expired(normalized_slot)

        self._refresh_for_seconds(
            function_id=normalized_function_id,
            node_id=normalized_primary_id,
            decision_slot=normalized_slot,
            seconds=normalized_primary_seconds,
        )
        for node_id in normalized_backup_ids:
            self._refresh_for_seconds(
                function_id=normalized_function_id,
                node_id=node_id,
                decision_slot=normalized_slot,
                seconds=normalized_backup_seconds,
            )

    def hot_node_ids(self, function_id: int, *, slot: int) -> frozenset[int]:
        """返回指定 VNF 在给定时隙仍保持温热的全部节点。"""

        normalized_function_id = _validate_nonnegative_integer(
            function_id,
            "function_id",
        )
        normalized_slot = _validate_nonnegative_integer(slot, "slot")
        self._purge_expired(normalized_slot)
        return frozenset(
            node_id
            for (stored_function_id, node_id), expiry_slot
            in self._expiry_slots.items()
            if stored_function_id == normalized_function_id
            and normalized_slot < expiry_slot
        )

    def remove_failed_nodes(self, failed_node_ids: Collection[int]) -> None:
        """立即删除故障节点上的全部 VNF 状态，不等待原保留时间结束。"""

        normalized_failed_ids = frozenset(
            _validate_nonnegative_integer(node_id, "failed node ID")
            for node_id in failed_node_ids
        )
        self._expiry_slots = {
            key: expiry_slot
            for key, expiry_slot in self._expiry_slots.items()
            if key[1] not in normalized_failed_ids
        }

    def reset(self) -> None:
        """开始新 Episode 时清空全部保留状态。"""

        self._expiry_slots.clear()

    def _refresh_for_seconds(
        self,
        *,
        function_id: int,
        node_id: int,
        decision_slot: int,
        seconds: float,
    ) -> None:
        """把正秒数换算为到期时隙；零秒不会创建或缩短保留。"""

        if seconds == 0.0:
            return
        retained_slot_count = math.ceil(seconds / self.slot_seconds)
        # 加一表示保留从决策后的下一个完整快时隙开始。
        new_expiry_slot = decision_slot + retained_slot_count + 1
        key = (function_id, node_id)
        self._expiry_slots[key] = max(
            self._expiry_slots.get(key, new_expiry_slot),
            new_expiry_slot,
        )

    def _purge_expired(self, slot: int) -> None:
        """删除在当前时隙开始前已经到期的记录，避免状态长期累积。"""

        self._expiry_slots = {
            key: expiry_slot
            for key, expiry_slot in self._expiry_slots.items()
            if slot < expiry_slot
        }
