"""测试只使用已观测历史的慢周期负载预测器。"""

import math

import pytest

from src.workload_prediction import (
    HistoricalWorkloadPredictor,
)


def test_forecast_uses_only_recent_history() -> None:
    """超过回看长度的旧数据不应影响当前预测。"""

    predictor = HistoricalWorkloadPredictor(
        lookback_slots=3,
        baseline_request_rate=1.0,
    )

    result = predictor.predict([9, 1, 2, 3])

    assert result.mean_requests == pytest.approx(2.0)
    assert result.peak_requests == 3.0
    assert result.normalized_trend == pytest.approx(
        2.0 / 3.0
    )


def test_future_suffix_cannot_change_current_forecast() -> None:
    """历史前缀相同时，未来轨迹不同也不能改变当前预测。"""

    predictor = HistoricalWorkloadPredictor(
        lookback_slots=3,
        baseline_request_rate=1.0,
    )
    first_trace = [0, 1, 3, 99, 99]
    second_trace = [0, 1, 3, 0, 0]

    first = predictor.predict(first_trace[:3])
    second = predictor.predict(second_trace[:3])

    assert first == second


def test_empty_history_uses_configured_baseline() -> None:
    """仿真刚开始没有历史数据时使用配置基准值。"""

    result = HistoricalWorkloadPredictor(
        lookback_slots=4,
        baseline_request_rate=2.0,
    ).predict([])

    assert result.mean_requests == 2.0
    assert result.peak_requests == 2.0
    assert result.normalized_trend == 0.0


def test_single_observation_has_zero_trend() -> None:
    """只有一个历史点时没有足够信息判断上升或下降。"""

    result = HistoricalWorkloadPredictor(
        lookback_slots=3,
        baseline_request_rate=1.0,
    ).predict([4])

    assert result.mean_requests == 4.0
    assert result.peak_requests == 4.0
    assert result.normalized_trend == 0.0


@pytest.mark.parametrize(
    ("lookback_slots", "baseline_request_rate", "message"),
    [
        (0, 1.0, "回看时隙数"),
        (3, -1.0, "基准请求率"),
        (3, math.inf, "基准请求率"),
    ],
)
def test_invalid_predictor_configuration_is_rejected(
    lookback_slots: int,
    baseline_request_rate: float,
    message: str,
) -> None:
    """无效配置应在创建预测器时尽早报告。"""

    with pytest.raises(ValueError, match=message):
        HistoricalWorkloadPredictor(
            lookback_slots=lookback_slots,
            baseline_request_rate=(
                baseline_request_rate
            ),
        )


def test_negative_observed_request_count_is_rejected() -> None:
    """请求数量不能为负，避免错误数据进入状态编码。"""

    predictor = HistoricalWorkloadPredictor(
        lookback_slots=3,
        baseline_request_rate=1.0,
    )

    with pytest.raises(ValueError, match="历史请求数"):
        predictor.predict([1, -1, 2])
