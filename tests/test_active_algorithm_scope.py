"""检查 DPPO 活动分支是否仍混入已经退役的算法资产。"""

from pathlib import Path

from src.config import load_config


# 拆开拼接旧算法名称，避免本测试文件的名字或常量本身被误判。
RETIRED_TOKEN = "d" + "dqn"


def test_active_tree_has_no_retired_named_files() -> None:
    """源码、测试和结果目录中不应再有以旧算法命名的文件。"""

    root = Path(__file__).resolve().parents[1]
    checked_roots = (root / "src", root / "tests", root / "results")
    offenders = [
        path.relative_to(root).as_posix()
        for checked_root in checked_roots
        if checked_root.exists()
        for path in checked_root.rglob("*")
        # Python 字节码只是本机测试缓存，不属于活动源码或实验资产。
        if path.is_file()
        and "__pycache__" not in path.parts
        and RETIRED_TOKEN in path.name.lower()
    ]

    assert offenders == []


def test_config_has_no_retired_sections() -> None:
    """活动配置的顶层键不能继续暴露旧算法训练或评估入口。"""

    config = load_config("configs/debug.yaml")

    assert all(RETIRED_TOKEN not in key.lower() for key in config)
