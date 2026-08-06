"""
config.py

本文件负责读取 YAML 配置文件。

例如：
    configs/debug.yaml

读取后，YAML 内容会被转换成 Python 字典。
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    读取 YAML 配置文件。

    Parameters
    ----------
    config_path:
        配置文件路径，例如 "configs/debug.yaml"。

    Returns
    -------
    dict[str, Any]:
        读取后的配置字典。

    Raises
    ------
    FileNotFoundError:
        当配置文件不存在时抛出。

    ValueError:
        当配置文件为空，或者最外层不是字典时抛出。
    """

    # 将字符串路径转换成 Path 对象，
    # 便于后续检查文件是否存在和打开文件。
    path = Path(config_path)

    # 检查配置文件是否存在。
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    # 使用 UTF-8 编码打开文件，避免中文乱码。
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 空 YAML 文件读取后会得到 None。
    if config is None:
        raise ValueError(f"配置文件为空：{path}")

    # 本项目要求 YAML 最外层必须是键值对结构。
    if not isinstance(config, dict):
        raise ValueError("配置文件最外层必须是字典结构。")

    return config