"""测试主算法使用的三成本加单一硬违约奖励。"""

import math

import pytest

from src.rl_reward import RLCostRewardBreakdown, RLWindowCostMetrics, calculate_cost_reward


def test_reward_is_normalized_cost_plus_violation() -> None:
    """奖励只包含归一化总成本和一个二值违约项。"""

    result = calculate_cost_reward(
        metrics=RLWindowCostMetrics(20.0, 10.0, 5.0, True),
        maximum_window_cost=100.0,
    )

    assert isinstance(result, RLCostRewardBreakdown)
    assert result.total_cost == 35.0
    assert result.normalized_total_cost == pytest.approx(0.35)
    assert result.violation_penalty == 1.0
    assert result.reward == pytest.approx(-1.35)


def test_no_request_warm_cost_is_still_charged() -> None:
    """没有请求时，主动保温产生的运行成本仍需计入奖励。"""

    result = calculate_cost_reward(
        RLWindowCostMetrics(12.0, 0.0, 0.0, False),
        maximum_window_cost=100.0,
    )

    assert result.reward == pytest.approx(-0.12)


def test_total_cost_is_clipped_before_penalty() -> None:
    """极端成本截断到 1 后，再叠加独立违约惩罚。"""

    result = calculate_cost_reward(
        RLWindowCostMetrics(200.0, 0.0, 0.0, True),
        maximum_window_cost=100.0,
    )

    assert result.total_cost == 200.0
    assert result.normalized_total_cost == 1.0
    assert result.reward == -2.0


@pytest.mark.parametrize("invalid_cost", [-1.0, math.inf, math.nan])
def test_invalid_window_cost_is_rejected(invalid_cost: float) -> None:
    """任何成本分量都必须是非负有限值。"""

    with pytest.raises(ValueError, match="窗口成本"):
        RLWindowCostMetrics(invalid_cost, 0.0, 0.0, False)


@pytest.mark.parametrize(
    "maximum_window_cost",
    [0.0, -1.0, math.inf, math.nan],
)
def test_invalid_maximum_window_cost_is_rejected(
    maximum_window_cost: float,
) -> None:
    """归一化分母必须是正有限值。"""

    with pytest.raises(ValueError, match="窗口最大成本"):
        calculate_cost_reward(
            RLWindowCostMetrics(1.0, 0.0, 0.0, False),
            maximum_window_cost=maximum_window_cost,
        )
