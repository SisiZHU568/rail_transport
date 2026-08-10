"""使用确定性零向量验证 DPPO 慢环境的一次完整执行闭环。"""

from pathlib import Path

import numpy as np

from src.config import load_config
from src.dppo_scenario import build_dppo_environment


def main() -> None:
    """构造配置化场景，执行一个慢窗口并打印可 review 的关键结果。"""

    project_root = Path(__file__).resolve().parent
    environment = build_dppo_environment(
        load_config(project_root / "configs" / "debug.yaml")
    )
    state = environment.reset()
    action = np.zeros(environment.dimensions.action_dim, dtype=np.float32)
    next_state, reward, terminated, truncated, info = environment.step(action)

    print("=" * 72)
    print("DPPO 双时间尺度环境单步演示")
    print("=" * 72)
    print(f"状态维度：{state.shape[0]}")
    print(f"动作维度：{action.shape[0]}")
    print(f"窗口：{info['window_start_slot']} -> {info['window_end_slot']}")
    print(f"投影成功：{info['projection_result'].success}")
    print(f"最终部署：{info['final_candidate_map']}")
    print(f"奖励：{reward:.6f}")
    print(f"下一状态维度：{next_state.shape[0]}")
    print(f"Episode 结束：{terminated}")
    print(f"截断：{truncated}")


if __name__ == "__main__":
    main()
