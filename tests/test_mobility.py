"""
test_mobility.py

测试列车移动模型。
"""

import pytest

from src.config import load_config
from src.mobility import TrainMobilityModel
from src.topology import build_linear_topology


def build_test_model() -> TrainMobilityModel:
    """
    创建供多个测试重复使用的列车移动模型。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    return TrainMobilityModel(
        topology=topology,
        initial_position_m=0.0,
        speed_mps=69.44,
        slot_seconds=1.0,
    )


def test_initial_train_state() -> None:
    """
    测试列车初始状态。
    """

    model = build_test_model()
    state = model.reset()

    assert state.time_slot == 0
    assert state.position_m == 0.0
    assert state.serving_mec == 0
    assert state.next_mec == 1


def test_train_moves_one_slot() -> None:
    """
    测试列车经过一个时隙后的新位置。
    """

    model = build_test_model()
    model.reset()

    state = model.step()

    assert state.time_slot == 1
    assert state.position_m == pytest.approx(69.44)
    assert state.serving_mec == 0


def test_train_handover() -> None:
    """
    测试列车能否从 MEC-1 自动切换到 MEC-2。

    MEC-1 和 MEC-2 的切换边界为 1000 米。

    这里将列车速度设为 600 m/s：

    时隙1：600米，仍连接 MEC-1；
    时隙2：1200米，应连接 MEC-2。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    model = TrainMobilityModel(
        topology=topology,
        initial_position_m=0.0,
        speed_mps=600.0,
        slot_seconds=1.0,
    )

    model.reset()

    state_1 = model.step()
    state_2 = model.step()

    assert state_1.position_m == pytest.approx(600.0)
    assert state_1.serving_mec == 0

    assert state_2.position_m == pytest.approx(1200.0)
    assert state_2.serving_mec == 1
    assert state_2.next_mec == 2


def test_train_does_not_exceed_route_end() -> None:
    """
    测试列车不会越过线路终点。
    """

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)

    model = TrainMobilityModel(
        topology=topology,
        initial_position_m=7900.0,
        speed_mps=500.0,
        slot_seconds=1.0,
    )

    model.reset()
    state = model.step()

    assert state.position_m == 8000.0
    assert model.finished is True
    assert state.serving_mec == 4