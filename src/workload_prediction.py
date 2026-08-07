"""提供只依赖已观测请求历史的慢周期负载预测。"""

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WorkloadForecast:
    """保存下一慢周期使用的三个负载预测特征。"""

    mean_requests: float
    peak_requests: float
    normalized_trend: float


class HistoricalWorkloadPredictor:
    """只接收已经观测到的请求前缀，禁止访问未来真实轨迹。"""

    def __init__(
        self,
        lookback_slots: int,
        baseline_request_rate: float,
    ) -> None:
        """配置历史回看长度和无历史数据时使用的基准请求率。"""

        if lookback_slots <= 0:
            raise ValueError(
                "回看时隙数必须大于0。"
            )

        if (
            not math.isfinite(
                baseline_request_rate
            )
            or baseline_request_rate < 0
        ):
            raise ValueError(
                "基准请求率必须是非负有限值。"
            )

        self.lookback_slots = lookback_slots
        self.baseline_request_rate = float(
            baseline_request_rate
        )

    def predict(
        self,
        observed_request_counts: Sequence[int],
    ) -> WorkloadForecast:
        """根据已完成快时隙的请求数量生成预测特征。"""

        if any(
            request_count < 0
            for request_count
            in observed_request_counts
        ):
            raise ValueError(
                "历史请求数不能小于0。"
            )

        # Episode 刚开始时没有历史数据，使用配置值而不是偷看未来轨迹。
        if not observed_request_counts:
            return WorkloadForecast(
                mean_requests=(
                    self.baseline_request_rate
                ),
                peak_requests=(
                    self.baseline_request_rate
                ),
                normalized_trend=0.0,
            )

        # 只截取已观测历史的末尾窗口；预测器不接收 workload 对象，
        # 因而无法按未来时隙编号查询真实请求数量。
        window = list(
            observed_request_counts[
                -self.lookback_slots:
            ]
        )
        mean_requests = sum(window) / len(window)
        peak_requests = float(max(window))

        if len(window) == 1:
            trend = 0.0
        else:
            # 用窗口峰值缩放首尾变化，并限制到 [-1, 1]，
            # 防止趋势特征因突发请求产生异常大的数值。
            denominator = max(
                peak_requests,
                1.0,
            )
            trend = (
                window[-1] - window[0]
            ) / denominator

        return WorkloadForecast(
            mean_requests=float(mean_requests),
            peak_requests=peak_requests,
            normalized_trend=float(
                max(-1.0, min(trend, 1.0))
            ),
        )
