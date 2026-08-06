"""
run_rl_environment_demo.py

演示强化学习环境的reset()和step()接口。

本程序暂时不训练神经网络，
而是使用一个可解释的人工策略：

    高风险窗口：HOT
    普通时段：DYNAMIC

用于验证：

1. 状态向量；
2. 动作输入；
3. 奖励输出；
4. MEC切换决策边界；
5. Episode终止。
"""

from pathlib import Path

from src.config import load_config
from src.entities import (
    ServerlessFunction,
    ServicePriority,
    SFCType,
)
from src.failure_risk_prediction import (
    build_windowed_failure_risk_provider,
)
from src.mobility import TrainMobilityModel
from src.network import build_linear_mec_network
from src.redundancy_placement import (
    build_reliability_aware_replica_planner,
)
from src.risk_aware_failure_process import (
    build_windowed_markov_failure_process,
)
from src.rl_reward import (
    RLRewardWeights,
)
from src.runtime_reliability import (
    SingleReplicaPlanner,
)
from src.slow_timescale_rl_env import (
    SlowControlAction,
    SlowTimescaleRLEnvironment,
)
from src.topology import build_linear_topology
from src.workload import DeterministicWorkload


def build_demo_functions() -> list[
    ServerlessFunction
]:
    """
    创建三函数故障诊断SFC。
    """

    return [
        ServerlessFunction(
            function_id=0,
            name="数据清洗",
            memory_mb=256.0,
            cpu_cycles_per_request=20.0,
            image_size_mb=80.0,
            warm_exec_time_ms=15.0,
            cold_start_time_ms=300.0,
            output_ratio=0.7,
        ),
        ServerlessFunction(
            function_id=1,
            name="特征提取",
            memory_mb=512.0,
            cpu_cycles_per_request=40.0,
            image_size_mb=150.0,
            warm_exec_time_ms=30.0,
            cold_start_time_ms=500.0,
            output_ratio=0.4,
        ),
        ServerlessFunction(
            function_id=2,
            name="异常检测",
            memory_mb=768.0,
            cpu_cycles_per_request=60.0,
            image_size_mb=300.0,
            warm_exec_time_ms=50.0,
            cold_start_time_ms=800.0,
            output_ratio=0.1,
        ),
    ]


def build_demo_sfc() -> SFCType:
    """
    创建强化学习演示SFC。
    """

    return SFCType(
        sfc_id=0,
        name="高可靠列车设备故障诊断",
        function_ids=[0, 1, 2],
        deadline_ms=1500.0,
        reliability_target=0.99,
        priority=ServicePriority.CRITICAL,
    )


def build_environment(
    config: dict,
) -> SlowTimescaleRLEnvironment:
    """
    根据配置构建强化学习环境。
    """

    topology = build_linear_topology(
        config
    )

    network = build_linear_mec_network(
        config=config,
        topology=topology,
    )

    mobility_model = TrainMobilityModel(
        topology=topology,
        initial_position_m=float(
            config["train"][
                "initial_position_m"
            ]
        ),
        speed_mps=float(
            config["train"]["speed_mps"]
        ),
        slot_seconds=float(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
    )

    workload = DeterministicWorkload(
        request_trace=(
            config[
                "integrated_simulation"
            ]["request_trace"]
        ),
        repeat=bool(
            config[
                "integrated_simulation"
            ]["repeat_request_trace"]
        ),
    )

    reward_config = config[
        "rl_environment"
    ]["reward_weights"]

    def failure_process_builder(
        seed: int,
    ):
        return (
            build_windowed_markov_failure_process(
                config=config,
                topology=topology,
                random_seed=seed,
            )
        )

    return SlowTimescaleRLEnvironment(
        topology=topology,
        network=network,
        mobility_model=mobility_model,
        workload=workload,
        functions=build_demo_functions(),
        sfc=build_demo_sfc(),
        single_replica_planner=(
            SingleReplicaPlanner()
        ),
        redundant_replica_planner=(
            build_reliability_aware_replica_planner(
                config
            )
        ),
        failure_process_builder=(
            failure_process_builder
        ),
        failure_risk_provider=(
            build_windowed_failure_risk_provider(
                config
            )
        ),
        slow_period_slots=int(
            config["two_timescale"][
                "slow_period_slots"
            ]
        ),
        handover_hot_window_s=float(
            config["two_timescale"][
                "handover_hot_window_s"
            ]
        ),
        input_size_mb_per_request=float(
            config[
                "integrated_simulation"
            ][
                "input_size_mb_per_request"
            ]
        ),
        slot_seconds=float(
            config["simulation"][
                "fast_slot_seconds"
            ]
        ),
        failover_delay_ms_per_function=float(
            config["runtime_failure"][
                "failover_delay_ms_per_function"
            ]
        ),
        reward_weights=RLRewardWeights(
            delay=float(
                reward_config["delay"]
            ),
            memory=float(
                reward_config["memory"]
            ),
            cold_start=float(
                reward_config["cold_start"]
            ),
            sla_violation=float(
                reward_config[
                    "sla_violation"
                ]
            ),
            reconfiguration=float(
                reward_config[
                    "reconfiguration"
                ]
            ),
        ),
        default_seed=int(
            config["rl_environment"][
                "default_seed"
            ]
        ),
        return_result_to_source=True,
    )


def select_rule_action(
    observation_info: dict,
    high_risk_threshold: float,
) -> SlowControlAction:
    """
    演示使用的人工动作选择规则。

    高风险时：
        全热备。

    普通时段：
        轨迹感知动态主备。
    """

    if (
        observation_info[
            "predicted_failure_risk"
        ]
        >= high_risk_threshold
    ):
        return SlowControlAction.HOT

    return SlowControlAction.DYNAMIC


def main() -> None:
    """
    程序入口。
    """

    project_root = (
        Path(__file__).resolve().parent
    )

    config = load_config(
        project_root
        / "configs"
        / "debug.yaml"
    )

    environment = build_environment(
        config
    )

    state, observation_info = (
        environment.reset()
    )

    threshold = float(
        config["two_timescale"][
            "high_failure_risk_threshold"
        ]
    )

    print("=" * 72)
    print("双时间尺度强化学习环境演示")
    print("=" * 72)

    print(
        f"状态维度："
        f"{environment.state_dim}"
    )

    print(
        f"动作数量："
        f"{environment.action_count}"
    )

    print(
        f"Episode随机种子："
        f"{observation_info['episode_seed']}"
    )

    print(
        f"快时隙总数："
        f"{observation_info['total_fast_slots']}"
    )

    total_reward = 0.0
    decision_number = 0

    terminated = False

    while not terminated:
        decision_number += 1

        action = select_rule_action(
            observation_info=(
                observation_info
            ),
            high_risk_threshold=(
                threshold
            ),
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            step_info,
        ) = environment.step(
            int(action)
        )

        metrics = step_info["metrics"]

        reward_breakdown = (
            step_info["reward_breakdown"]
        )

        # 先取出需要打印的数据。
        #
        # 不要在普通 f-string 中将字典索引或属性访问拆成多行，
        # 否则 Python 3.10 会出现 unterminated string literal。
        predicted_failure_risk = float(
            observation_info[
                "predicted_failure_risk"
            ]
        )

        average_delay_ms = float(
            metrics.average_successful_delay_ms
        )

        average_memory_mb = float(
            metrics.average_active_memory_mb
        )

        cold_start_delay_ms = float(
            metrics.total_cold_start_delay_ms
        )

        weighted_cost = float(
            reward_breakdown.weighted_cost
        )

        print("\n" + "=" * 72)
        print(
            f"慢决策 {decision_number}"
        )
        print("=" * 72)

        print(
            f"  决策起始时隙："
            f"{step_info['window_start_slot']}"
        )

        print(
            f"  决策结束时隙："
            f"{step_info['window_end_slot']}"
        )

        print(
            f"  窗口长度："
            f"{step_info['window_length']}"
        )

        print(
            f"  窗口结束原因："
            f"{step_info['boundary_reason']}"
        )

        print(
            f"  预测故障风险："
            f"{predicted_failure_risk:.3f}"
        )

        print(
            f"  选择动作："
            f"{action.name}"
        )

        print(
            f"  窗口请求数量："
            f"{metrics.total_requests}"
        )

        print(
            f"  请求成功率："
            f"{metrics.request_success_rate:.6f}"
        )

        print(
            f"  SLA违反率："
            f"{metrics.sla_violation_rate:.6f}"
        )

        print(
            f"  平均成功时延："
            f"{average_delay_ms:.3f} ms"
        )

        print(
            f"  平均活动内存："
            f"{average_memory_mb:.3f} MB"
        )

        print(
            f"  冷启动总时延："
            f"{cold_start_delay_ms:.3f} ms"
        )

        print(
            f"  归一化成本："
            f"{weighted_cost:.6f}"
        )

        print(
            f"  当前奖励："
            f"{reward:.6f}"
        )

        total_reward += reward

        if truncated:
            raise RuntimeError(
                "当前环境不应出现truncated=True。"
            )

        state = next_state

        observation_info = (
            step_info["next_observation"]
        )

    print("\n" + "=" * 72)
    print("强化学习Episode结束")
    print("=" * 72)

    print(
        f"慢决策总次数："
        f"{decision_number}"
    )

    print(
        f"Episode累计奖励："
        f"{total_reward:.6f}"
    )

    print(
        f"终止状态进度："
        f"{state[0]:.3f}"
    )


if __name__ == "__main__":
    main()