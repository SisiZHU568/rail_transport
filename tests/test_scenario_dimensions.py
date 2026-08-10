"""测试 DPPO 的规模描述对象和配置驱动场景构造。"""

from copy import deepcopy

import pytest

from src.config import load_config
from src.dppo_scenario import build_rl_functions, build_rl_sfc
from src.scenario_dimensions import ScenarioDimensions
from src.topology import build_linear_topology


@pytest.mark.parametrize(
    ("mec_ids", "function_ids", "state_dim", "action_dim"),
    [
        ((0, 1, 2), (0, 1), 58, 14),
        ((0, 1, 2, 3, 4), (0, 1, 2), 78, 27),
        (tuple(range(8)), (0, 1, 2), 102, 36),
    ],
)
def test_dimensions_follow_configured_scale(
    mec_ids: tuple[int, ...],
    function_ids: tuple[int, ...],
    state_dim: int,
    action_dim: int,
) -> None:
    """改变 MEC 或 VNF 数量时，维度必须自动重新计算。"""

    dimensions = ScenarioDimensions(
        mec_node_ids=mec_ids,
        cloud_node_id=len(mec_ids),
        function_ids=function_ids,
    )

    assert dimensions.compute_node_ids == mec_ids + (len(mec_ids),)
    assert dimensions.mec_count == len(mec_ids)
    assert dimensions.compute_node_count == len(mec_ids) + 1
    assert dimensions.function_count == len(function_ids)
    assert dimensions.state_dim == state_dim
    assert dimensions.action_dim == action_dim


def test_dimensions_reject_duplicate_or_insufficient_nodes() -> None:
    """节点编号必须唯一，并且至少能承载三个副本。"""

    with pytest.raises(ValueError, match="计算节点 ID 必须唯一"):
        ScenarioDimensions((0, 1, 2), 2, (0, 1))

    with pytest.raises(ValueError, match="至少需要三个计算节点"):
        ScenarioDimensions((0,), 1, (0,))

    with pytest.raises(ValueError, match="计算节点 ID 必须是非负整数"):
        ScenarioDimensions((True, 2), 3, (0,))


def test_config_builders_preserve_function_and_sfc_order() -> None:
    """配置中的 VNF 列表和 SFC 链顺序不能在构造时被重新排序。"""

    config = load_config("configs/debug.yaml")
    functions = build_rl_functions(config)
    sfc = build_rl_sfc(config)

    expected_function_ids = tuple(
        int(item["function_id"])
        for item in config["rl_scenario"]["functions"]
    )
    expected_chain = tuple(
        int(function_id)
        for function_id in config["rl_scenario"]["sfc"]["function_ids"]
    )

    assert tuple(function.function_id for function in functions) == expected_function_ids
    assert tuple(sfc.function_ids) == expected_chain


def test_dimensions_are_built_once_from_topology_and_functions() -> None:
    """真实场景应直接生成统一维度对象，不让下游模块重复猜测顺序。"""

    config = load_config("configs/debug.yaml")
    topology = build_linear_topology(config)
    functions = build_rl_functions(config)

    dimensions = ScenarioDimensions.from_scenario(topology, functions)

    assert dimensions.mec_node_ids == tuple(
        site.node.node_id for site in topology.sites
    )
    assert dimensions.cloud_node_id == topology.cloud_node.node_id
    assert dimensions.function_ids == tuple(
        function.function_id for function in functions
    )


def test_config_builders_reject_duplicate_function_ids() -> None:
    """重复的 VNF ID 会让动作切片含义不唯一，必须提前拒绝。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["rl_scenario"]["functions"][1]["function_id"] = 0

    with pytest.raises(ValueError, match="VNF ID.*唯一"):
        build_rl_functions(config)


def test_config_builders_reject_fractional_function_ids() -> None:
    """不能把小数 ID 静默截断成整数，否则会改变 VNF 身份。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["rl_scenario"]["functions"][0]["function_id"] = 0.5

    with pytest.raises(ValueError, match="VNF ID 必须是非负整数"):
        build_rl_functions(config)


def test_config_builders_reject_mismatched_sfc_chain() -> None:
    """SFC 链必须且只能引用配置中声明的全部 VNF。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["rl_scenario"]["sfc"]["function_ids"] = [0, 1, 99]

    with pytest.raises(ValueError, match="SFC.*VNF"):
        build_rl_sfc(config)


def test_config_builders_reject_sfc_order_different_from_action_order() -> None:
    """状态、动作和 SFC 执行必须对同一 VNF 顺序达成一致。"""

    config = deepcopy(load_config("configs/debug.yaml"))
    config["rl_scenario"]["sfc"]["function_ids"] = [1, 0, 2]

    with pytest.raises(ValueError, match="SFC.*顺序"):
        build_rl_sfc(config)
