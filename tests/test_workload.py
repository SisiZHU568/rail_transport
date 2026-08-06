
"""
test_workload.py

测试确定性请求负载模型。
"""

import pytest

from src.workload import DeterministicWorkload


def test_non_repeating_workload() -> None:
    """
    不循环请求轨迹时，
    超过轨迹长度后的请求数量应为0。
    """

    workload = DeterministicWorkload(
        request_trace=[0, 1, 2],
        repeat=False,
    )

    assert workload.request_count(0) == 0
    assert workload.request_count(1) == 1
    assert workload.request_count(2) == 2
    assert workload.request_count(3) == 0
    assert workload.request_count(100) == 0


def test_repeating_workload() -> None:
    """
    循环轨迹 [0, 1, 2] 应不断重复。
    """

    workload = DeterministicWorkload(
        request_trace=[0, 1, 2],
        repeat=True,
    )

    assert workload.request_count(0) == 0
    assert workload.request_count(1) == 1
    assert workload.request_count(2) == 2
    assert workload.request_count(3) == 0
    assert workload.request_count(4) == 1
    assert workload.request_count(5) == 2


def test_invalid_workload_trace() -> None:
    """
    空轨迹和负请求数量都应报错。
    """

    with pytest.raises(ValueError):
        DeterministicWorkload(
            request_trace=[],
            repeat=False,
        )

    with pytest.raises(ValueError):
        DeterministicWorkload(
            request_trace=[0, -1, 2],
            repeat=False,
        )