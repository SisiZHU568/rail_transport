# Phase E 在线 PPO 训练就绪设计

## 目标

在不使用运行时补丁的前提下，让 Phase E 真实环境连续执行至少 32 个慢帧、完成 2 次 PPO 更新，并生成可恢复的在线 checkpoint、训练历史和失败审计。正式入口必须从规格绑定的 Phase E 预训练 checkpoint 与教师数据启动；随机初始化仍只用于 smoke 验证。

## 当前问题

DPPO 与环境的核心调用链已经存在，但尚未达到正式训练条件：

- `QueueStateManager` 的流量审计使用配置的绝对/相对容差，而残片保留和批次完成仍使用硬编码 `1e-9` bit。真实运行会积累大量亚 bit fragment，使 CLARABEL 在首次 PPO 更新前进入 `solver_error`。
- `run_phase_e_online_smoke.py` 能采样、执行和更新，但不保存 checkpoint、训练历史或验证结果，也不支持恢复训练。
- 仓库现有训练产物属于旧 checkpoint 与教师数据格式，不能绕过 Phase E 的规格哈希检查复用。

## 架构

改动分成两个顺序依赖的组件。

### 队列数值归一化边界

`QueueStateManager` 是唯一允许消化浮点残量的状态边界。优化器继续按 Mbit 求解，但提交端统一以 bit 为单位使用 `_flow_tolerance()`：

- 当一个 fragment 已发生有效消费，且未消费尾量不超过该次可用流量的容差时，把尾量并入最后一个成功的分配操作，而不是生成 residual fragment。
- 被并入的尾量沿原操作语义生成下一阶段 fragment、传输记录或完成事件，因此总输入等效量保持守恒。
- 高于容差的尾量继续保留为 residual fragment。
- 批次完成判断使用相同的绝对/相对容差；落入容差后把完成量规范化为批次总量。
- 超量、无有效目标、版本过期和超过容差的流量不平衡仍然拒绝提交，不引入静默回退。

`FastResourceOptimizer` 只读取归一化快照，不负责修改队列或丢弃小队列。这样可避免在求解器与提交端形成两套流量真值。

### 正式在线训练入口

新增 `run_phase_e_online_training.py`，复用现有 `PhaseESlowFrameEnvironment`、`PhaseEOnlineTrainer`、观察适配器和安全部署控制器。入口负责：

- 强制同时提供 `--pretrained-checkpoint` 与 `--teacher-dataset`，并校验观察规格、动作规格和数据维度。
- 按配置或 CLI 指定的慢帧数运行真实双时间尺度环境。
- 每 16 个完整慢帧提交一次 PPO 更新；不足一个完整 rollout 的尾部不跨进程保存。
- 每次更新后原子保存 `dppo_phase_e_last.pt`，验证指标改善时保存 `dppo_phase_e_best.pt`。
- 追加写入 `training_history.csv`，记录帧范围、奖励、违约率、服务缺口、成本、策略损失、价值损失、KL、clip fraction、梯度范数和 BC 权重。
- 将内部失败写入 `failure_audit.json`，丢弃当前未提交 rollout，并以非零状态退出。
- 支持 `--resume-checkpoint`，恢复模型、价值网络、优化器、完成更新数、下一慢帧编号以及 Python、NumPy、Torch CPU/CUDA 随机数状态。

在线 checkpoint 使用独立的 `phase-e-online-v1` 格式，包含观察/动作规格哈希、环境配置指纹和训练计数。恢复时不接受配置或规格不一致的 checkpoint。

## 数据流

每个慢帧按固定顺序执行：

1. 从队列、生命周期、故障、网络和上一慢帧结果编码观察。
2. DPPO 采样完整去噪链和无界动作，再映射为有界部署评分。
3. 安全部署解码器生成生命周期计划，并通过版本校验原子提交。
4. 每个快时隙执行无线、计算和有线凸优化，再由队列提交端应用分配并归一化数值尾量。
5. 完整慢帧计算奖励并记录 transition。
6. 满 16 个慢帧后计算 GAE，执行 DPPO clipped policy loss、价值网络更新、KL early stop 和早期教师 BC。
7. 写入训练历史与 checkpoint，然后开始下一事务 rollout。

## 错误处理

- `FAST_SOLVER_FAILURE`、`STALE_SNAPSHOT`、`FLOW_CONSERVATION_VIOLATION` 等内部错误不会产生 transition 或 PPO 更新。当前未提交 rollout 被丢弃，失败上下文写入审计文件，进程失败退出。
- 配置、模型格式、规格哈希或数据维度不匹配在环境启动前失败，且不创建训练输出。
- checkpoint 使用临时文件加原子替换，避免中断留下半写文件。
- 恢复仅从完整 PPO 更新边界继续，不序列化半截 rollout。
- `OPTIMAL_INACCURATE` 只有在全部有限性与物理残差审计通过后才可接受。

## 测试

### 队列单元测试

- 覆盖低于、等于和高于绝对/相对容差的残量。
- 验证中间阶段、跨链路转发和最终阶段完成都保持输入等效量守恒。
- 验证容差内完成量被规范化为批次总量，容差外残量继续留在队列。

### 快层回归测试

- 构造连续部分消费形成的数值尾量，确认快照不会积累亚 bit fragment。
- 验证真实 CLARABEL 求解不再因该状态返回 `solver_error`。

### 训练持久化测试

- checkpoint 往返恢复策略、价值网络、优化器、更新计数和随机数状态。
- 规格或配置指纹不匹配时拒绝恢复。
- 内部失败只写审计文件，不写成功历史或新 checkpoint。

### 端到端验收

使用 CPU 和 debug 规模运行 32 个真实慢帧：

- 完成 2 次 PPO optimizer update。
- 内部失败数为 0。
- 生成并能重新加载 `dppo_phase_e_last.pt` 与 `dppo_phase_e_best.pt`。
- `training_history.csv` 包含 2 条完整更新记录。
- 运行现有全量测试，保持所有既有断言通过。

## 非目标

- 本阶段不宣称长时间训练已经收敛，也不确定论文最终超参数。
- 不恢复旧 `dppo-checkpoint-v1` 或 `dppo-online-checkpoint-v2` 兼容入口。
- 不在求解失败时增加启发式资源分配或随机动作回退。
- 不重构与 Phase E 在线训练无关的旧实验结果目录。

