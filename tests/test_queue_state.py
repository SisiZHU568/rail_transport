"""测试阶段 B 不可变队列数据模型和等效量换算。"""

from dataclasses import FrozenInstanceError

import pytest

from src.queue_state import (
    BatchRecord,
    CompletionEvent,
    InTransitRecord,
    QueueFragment,
    QueueSnapshot,
    StageFlowConfig,
)


def test_stage_flow_uses_input_equivalent_bits_as_single_truth() -> None:
    flow = StageFlowConfig(output_ratios=(0.5, 2.0, 0.25))

    assert flow.gamma(0) == pytest.approx(1.0)
    assert flow.gamma(1) == pytest.approx(0.5)
    assert flow.gamma(2) == pytest.approx(1.0)
    assert flow.physical_bits(1, 20.0) == pytest.approx(10.0)
    assert flow.equivalent_bits(1, 10.0) == pytest.approx(20.0)


@pytest.mark.parametrize("ratios", [(), (0.0,), (-1.0,), (float("inf"),)])
def test_stage_flow_rejects_invalid_output_ratios(ratios: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="output ratio"):
        StageFlowConfig(output_ratios=ratios)


def test_fragment_validates_stage_and_equivalent_amount() -> None:
    with pytest.raises(ValueError, match="stage_id"):
        QueueFragment("f", "b", 0, -2, 0, None, 1.0, 0)
    with pytest.raises(ValueError, match="input_equivalent_bits"):
        QueueFragment("f", "b", 0, 0, 0, None, 0.0, 0)


def test_in_transit_record_requires_exact_fragment_amount() -> None:
    fragment = QueueFragment("f", "b", 0, 1, 1, 2, 4.0, 4)

    with pytest.raises(ValueError, match="equivalent"):
        InTransitRecord("t", fragment, 0, 1, 7, 3, 4, 3.9)


def test_in_transit_fragment_matches_destination_and_arrival() -> None:
    wrong_destination = QueueFragment("f1", "b", 0, 1, 0, 2, 4.0, 4)
    wrong_arrival = QueueFragment("f2", "b", 0, 1, 1, 2, 4.0, 5)

    with pytest.raises(ValueError, match="destination"):
        InTransitRecord("t1", wrong_destination, 0, 1, 7, 3, 4, 4.0)
    with pytest.raises(ValueError, match="arrival"):
        InTransitRecord("t2", wrong_arrival, 0, 1, 7, 3, 4, 4.0)


def test_snapshot_and_nested_records_are_immutable() -> None:
    batch = BatchRecord("b", 0, 0.0, 5.0)
    fragment = QueueFragment("f", "b", 0, -1, None, None, 2.0, 0)
    completion = CompletionEvent("c", "b", 1, 2.0)
    snapshot = QueueSnapshot(
        version=0,
        current_slot=0,
        batches=(batch,),
        uplink_fragments=(fragment,),
        stage_fragments=(),
        in_transit=(),
        completion_events=(completion,),
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.current_slot = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fragment.input_equivalent_bits = 9.0  # type: ignore[misc]


def test_batch_and_completion_event_validate_time_boundaries() -> None:
    with pytest.raises(ValueError, match="deadline"):
        BatchRecord("b", 0, 2.0, 1.0)
    with pytest.raises(ValueError, match="completion_slot"):
        CompletionEvent("c", "b", -1, 0.0)
