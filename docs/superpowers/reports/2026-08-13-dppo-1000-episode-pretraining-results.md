# DPPO 1000 Episode 专家数据与 GPU 预训练结果

## 1. 运行环境

- Python 环境：`rail-dppo-gpu`
- PyTorch：2.11.0+cu128
- GPU：NVIDIA GeForce RTX 5060
- 预训练设备：CUDA
- 快时间尺度求解器：CLARABEL（CPU）
- 配置哈希：`f6a7808f14dca6ddbf5048bd6ace98dacd507c0235cf34a0b1281e8724a21815`

## 2. 1000 Episode 专家数据

本次使用当前教师规则生成 1000 episode，每个 episode 最多包含 2 个慢尺度决策点，共得到 2000 个教师候选。

| 教师 | 候选数 | 有效标签 | 诊断记录 | 接受率 |
|---|---:|---:|---:|---:|
| cost | 668 | 302 | 366 | 45.21% |
| reliability | 666 | 335 | 331 | 50.30% |
| balanced | 666 | 662 | 4 | 99.40% |

有效专家标签分区：

| 分区 | cost | reliability | balanced | 合计 |
|---|---:|---:|---:|---:|
| train | 202 | 225 | 508 | 935 |
| validation | 49 | 55 | 74 | 178 |
| test | 51 | 55 | 80 | 186 |

全部 1299 条保存标签的 `expert_action_feasible` 和 `final_feasible` 均为真。

## 3. Held-out 教师执行诊断

对 validation 和 test 中的 364 条标签重新构建场景并真实执行：

| 教师 | 样本数 | 原始可行率 | 投影修改率 | 快层成功率 | 平均奖励 | CLARABEL 平均耗时 |
|---|---:|---:|---:|---:|---:|---:|
| cost | 100 | 100% | 0% | 100% | -0.015479 | 5.08 ms |
| reliability | 110 | 100% | 0% | 100% | -0.012014 | 5.30 ms |
| balanced | 154 | 100% | 0% | 100% | -0.008900 | 5.00 ms |

教师数据门禁通过，CLARABEL 当前也没有形成明显耗时瓶颈。

## 4. GPU 专家预训练

### 200 步诊断

200 次 optimizer update 后：

- 验证损失：0.977777 降至 0.320568；
- 最佳点仍为第 200 步；
- 相对随机模型的 held-out 平均动作 MSE 只下降 6.59%；
- 原始动作可行率由 26.37% 提高到 31.87%；
- 诊断结论：`continuous_signal_not_learned`。

这说明 200 步时模型仍在学习，尚未达到进入在线训练所需的预训练门禁。

### 单变量延长到 1000 步

保持数据、网络结构、学习率、batch size、随机种子和诊断样本不变，只把 optimizer update 增加到 1000 次：

- 验证损失：0.977777 降至 0.161167；
- 最佳 checkpoint：第 1000 步；
- held-out 平均动作 MSE：随机模型 1.425625，预训练模型 1.058130；
- 相对平均 MSE 降幅：25.78%；
- 76.10% 的 held-out 记录 MSE 优于随机模型；
- 原始动作可行率：26.37% 提高到 46.43%；
- 投影成功率：100%；
- 诊断结论：`continuous_and_deployment_signal_learned`。

因此后续应使用 1000 步 checkpoint，不再使用 200 步 checkpoint。

## 5. 本地产物

这些运行产物位于 `results/dppo/`，已按项目规则忽略，不提交到 Git：

- 专家数据：`results/dppo/datasets/teacher_fix_1000_v1/`
- 教师诊断：`results/dppo/diagnostics/teacher_fix_1000_v1/`
- 1000 步 checkpoint：`results/dppo/pretraining/teacher_fix_1000_steps1000_v1/dppo_pretrained.pt`
- 训练曲线：`results/dppo/pretraining/teacher_fix_1000_steps1000_v1/pretraining_history.csv`
- 预训练诊断：`results/dppo/diagnostics/pretraining_teacher_fix_1000_steps1000_v1/`

1000 步 checkpoint SHA-256：

`82bb9100da2311d1c93ad266ca58d6fb9512239153729ee6a942a65c5349a181`

## 6. 下一步

专家数据和预训练门禁已经通过。下一阶段应以 1000 步 checkpoint 为起点，执行 DPPO 稳定性校准，选择正式在线训练使用的 PPO `clip_ratio`；校准通过后再开始扩大在线训练规模。
