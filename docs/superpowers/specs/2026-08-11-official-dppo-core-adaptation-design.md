# 官方 DPPO 核心适配设计

## 目标

将 `irom-princeton/dppo` 的 DPPO 核心训练公式适配到当前轨道交通
Serverless SFC 双时间尺度仿真中。保留已有轨道环境、状态编码、联合动作解释、
约束投影和快层执行，不引入官方项目面向机器人控制的外围工程。

## 范围

本阶段只实现并验证以下主链路：

1. 扩散策略生成完整去噪动作链；
2. 轨道慢尺度状态被包装成官方接口使用的状态序列；
3. 联合 SFC 动作被包装成长度为 1 的连续动作块；
4. 使用官方 DPPO 的去噪折扣和分阶段 PPO 裁剪计算策略损失；
5. 完成一次“环境交互—轨迹收集—策略更新”的冒烟运行。

本阶段不引入 MuJoCo、IsaacGym、Robomimic、Furniture-Bench、视频录制、
WandB、并行环境或机器人数据集。

## 来源和许可证

- 上游仓库：<https://github.com/irom-princeton/dppo>
- 主要参考文件：
  - `model/diffusion/diffusion_ppo.py`
  - `model/diffusion/diffusion_vpg.py`
  - `agent/finetune/train_ppo_diffusion_agent.py`
- 上游许可证：MIT License，Copyright (c) 2024 Intelligent Robot Motion Lab。

仓库内保存上游许可证和来源说明。适配代码文件头明确标注上游链接、原始文件
和“针对轨道 SFC 环境进行接口适配”，避免将适配实现误写为完全自研算法。

## 数据接口

官方实现的主要张量约定为：

- 状态：`(B, To, Do)`；
- 去噪动作链：`(B, K + 1, Ta, Da)`；
- 单步去噪概率：`(B, K, Ta, Da)`。

轨道环境的适配约定为：

- `To = 1`：每次使用一个慢尺度决策点状态；
- `Ta = 1`：每个慢时隙一次生成整套联合编排动作；
- `Do = state_encoder.output_dim`；
- `Da = action_space.dimension`。

`Do` 和 `Da` 均由场景配置动态计算。扩大 MEC、VNF 或 SFC 规模时，只改变配置
和编码维度，不在算法中写死节点数。

## 模块边界

### 新增官方核心适配模块

`src/dppo_official_core.py` 只负责官方 DPPO 数学核心：

- 根据去噪索引计算官方的指数插值裁剪系数；
- 对环境优势乘以去噪 MDP 折扣；
- 计算 clipped PPO policy loss、近似 KL 和 clip fraction；
- 返回结构化的损失与诊断指标。

该模块不读取 YAML、不创建轨道环境，也不解释 SFC 动作。

### 修改现有 DPPO 代理

`src/dppo.py` 继续负责：

- 冻结基础扩散策略并创建可训练副本；
- 采样去噪链并保存旧概率；
- 计算 GAE 和价值损失；
- 调用 `dppo_official_core` 完成策略更新。

现有固定裁剪系数将替换为官方分阶段裁剪系数。现有数值稳定 KL 计算和更新前
早停继续保留，因为它不改变官方优化目标，只避免无效的溢出更新。

### 保留轨道环境

`src/dppo_slow_timescale_env.py`、状态编码、动作空间、约束投影和快层执行逻辑
保持现有职责。适配发生在 DPPO 代理边界，不让轨道模块依赖机器人训练框架。

## 配置

在现有 DPPO 配置中增加两个官方裁剪调度参数：

- `clip_ratio_base`：最早微调去噪步骤的裁剪下限，默认 `0.001`；
- `clip_ratio_rate`：指数插值速率，默认 `3.0`。

现有 `clip_ratio` 仍表示最终去噪步骤的最大裁剪范围，
`denoising_discount` 对应官方 `gamma_denoising`。所有数值从 YAML 读取。

## 核心测试

只新增一个精简测试文件，覆盖四项行为：

1. 不同 `Do/Da` 下的状态、动作块和去噪链形状；
2. 官方裁剪调度从 `clip_ratio_base` 增长到 `clip_ratio`；
3. 官方策略损失可反向传播并产生有限梯度；
4. 真实轨道环境完成短 rollout 和一次 `DPPOAgent.update()`。

验收以该核心测试和直接受影响的原有 DPPO 测试通过为准，不在本任务中运行或扩展
无关模块测试。

## 实施原则

- 先写失败测试，再写最小实现；
- 添加面向初学者的必要中文注释；
- 不顺带重构无关模块；
- 每个可运行阶段单独提交并推送，便于 review 和回退。
