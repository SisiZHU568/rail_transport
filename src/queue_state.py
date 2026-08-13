"""阶段 B 的不可变队列、批次、在途和完成事件数据模型。"""

from dataclasses import dataclass
import math


def _positive_finite(value: float, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} 必须是正有限数。")


@dataclass(frozen=True)
class BatchRecord:
    """批次截止时间从进入系统起固定，经过各阶段时不会重置。"""

    batch_id: str
    service_id: int
    arrival_time: float
    absolute_deadline_time: float
    total_input_equivalent_bits: float = 1.0
    completed_input_equivalent_bits: float = 0.0
    completion_slot: int | None = None
    violation_recorded: bool = False

    def __post_init__(self) -> None:
        if not self.batch_id or self.service_id < 0:
            raise ValueError("batch_id 不能为空且 service_id 不能为负。")
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0.0:
            raise ValueError("arrival_time 必须是非负有限数。")
        if (
            not math.isfinite(self.absolute_deadline_time)
            or self.absolute_deadline_time < self.arrival_time
        ):
            raise ValueError("deadline 不能早于到达时间。")
        _positive_finite(
            self.total_input_equivalent_bits,
            "total_input_equivalent_bits",
        )
        if (
            not math.isfinite(self.completed_input_equivalent_bits)
            or self.completed_input_equivalent_bits < 0.0
            or self.completed_input_equivalent_bits
            > self.total_input_equivalent_bits + 1e-9
        ):
            raise ValueError("completed_input_equivalent_bits 超出批次总量。")
        if self.completion_slot is not None and self.completion_slot < 0:
            raise ValueError("completion_slot 不能为负。")


@dataclass(frozen=True)
class QueueFragment:
    """只保存原始输入等效 bit；物理数据量始终由阶段比例派生。"""

    fragment_id: str
    batch_id: str
    service_id: int
    stage_id: int
    location: int | None
    routing_target_node: int | None
    input_equivalent_bits: float
    available_slot: int

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.batch_id or self.service_id < 0:
            raise ValueError("片段 ID、批次 ID 和业务 ID 无效。")
        if self.stage_id < -1:
            raise ValueError("stage_id 只能是 -1（上行）或非负 VNF 编号。")
        if self.stage_id == -1 and self.location is not None:
            raise ValueError("上行片段不能预先指定部署位置。")
        if self.stage_id >= 0 and (self.location is None or self.location < 0):
            raise ValueError("阶段片段必须位于一个非负节点 ID。")
        if self.routing_target_node is not None and self.routing_target_node < 0:
            raise ValueError("routing_target_node 不能为负。")
        _positive_finite(self.input_equivalent_bits, "input_equivalent_bits")
        if self.available_slot < 0:
            raise ValueError("available_slot 不能为负。")


@dataclass(frozen=True)
class InTransitRecord:
    """创建后不可修改；到达时再依据最新快照决定是否清除目标绑定。"""

    transit_id: str
    fragment: QueueFragment
    source_node_id: int
    destination_node_id: int
    link_id: int
    departure_slot: int
    arrival_slot: int
    input_equivalent_bits: float

    def __post_init__(self) -> None:
        if not self.transit_id:
            raise ValueError("transit_id 不能为空。")
        if min(self.source_node_id, self.destination_node_id, self.link_id) < 0:
            raise ValueError("在途节点和链路 ID 不能为负。")
        if self.departure_slot < 0 or self.arrival_slot <= self.departure_slot:
            raise ValueError("arrival_slot 必须严格晚于 departure_slot。")
        if self.fragment.location != self.destination_node_id:
            raise ValueError("在途片段 location 必须等于 destination 节点。")
        if self.fragment.available_slot != self.arrival_slot:
            raise ValueError("在途片段 available_slot 必须等于 arrival_slot。")
        _positive_finite(self.input_equivalent_bits, "input_equivalent_bits")
        if not math.isclose(
            self.input_equivalent_bits,
            self.fragment.input_equivalent_bits,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("在途记录与片段的 equivalent 数量不一致。")


@dataclass(frozen=True)
class CompletionEvent:
    """最终 VNF 的输出在下一时隙边界才完成批次。"""

    event_id: str
    batch_id: str
    completion_slot: int
    input_equivalent_bits: float

    def __post_init__(self) -> None:
        if not self.event_id or not self.batch_id:
            raise ValueError("完成事件 ID 和批次 ID 不能为空。")
        if self.completion_slot < 0:
            raise ValueError("completion_slot 不能为负。")
        _positive_finite(self.input_equivalent_bits, "input_equivalent_bits")


@dataclass(frozen=True)
class QueueSnapshot:
    """队列状态所有者对外发布的完整只读快照。"""

    version: int
    current_slot: int
    batches: tuple[BatchRecord, ...]
    uplink_fragments: tuple[QueueFragment, ...]
    stage_fragments: tuple[QueueFragment, ...]
    in_transit: tuple[InTransitRecord, ...]
    completion_events: tuple[CompletionEvent, ...]

    def __post_init__(self) -> None:
        if self.version < 0 or self.current_slot < 0:
            raise ValueError("队列版本和当前时隙不能为负。")


@dataclass(frozen=True)
class StageFlowConfig:
    """用 VNF 输出比例生成各阶段的累计物理量比例。"""

    output_ratios: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.output_ratios:
            raise ValueError("output ratio 列表不能为空。")
        for ratio in self.output_ratios:
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError("output ratio 必须是正有限数。")

    @property
    def stage_count(self) -> int:
        return len(self.output_ratios)

    def gamma(self, stage_id: int) -> float:
        if stage_id < 0 or stage_id >= self.stage_count:
            raise ValueError("stage_id 超出 SFC 范围。")
        value = 1.0
        for ratio in self.output_ratios[:stage_id]:
            value *= ratio
        return value

    def physical_bits(self, stage_id: int, equivalent_bits: float) -> float:
        _positive_finite(equivalent_bits, "equivalent_bits")
        return self.gamma(stage_id) * equivalent_bits

    def equivalent_bits(self, stage_id: int, physical_bits: float) -> float:
        _positive_finite(physical_bits, "physical_bits")
        return physical_bits / self.gamma(stage_id)
