# DPPO GPU 冒烟训练设计

## 1. 目标

创建一个不影响现有 CPU 环境的独立 Conda 环境，并完成当前 v2 双时间尺度
主算法的一次最小 GPU 闭环：

```text
v2 仿真数据 → GPU 扩散预训练 → 稳定性校准 → GPU 在线 DPPO 更新
                                  ↓
                         CPU CLARABEL 快层调度
```

本阶段只证明主算法、GPU 和新快层能够一起运行，不进行正式论文训练，也不加入
基线或独立评估。

## 2. 环境隔离

新环境固定命名为 `rail-dppo-gpu`，使用 Python 3.12。它是干净环境，不从
Anaconda base 克隆，从而避免复制现有 CPU 版 PyTorch 和历史依赖。

环境中只安装项目当前闭环需要的依赖：

- 支持 RTX 5060 的官方 CUDA 版 PyTorch；
- NumPy、PyYAML、pytest 等项目运行依赖；
- CVXPY 1.6.x 和 CLARABEL；
- 当前代码实际导入所需的其他最小包。

安装后必须验证：

- `torch.cuda.is_available()` 为 `True`；
- PyTorch 能识别 `NVIDIA GeForce RTX 5060`；
- 一个张量运算真实发生在 `cuda:0`；
- CVXPY 的已安装求解器包含 `CLARABEL`；
- `pip check` 没有依赖冲突。

若 CUDA 版 PyTorch 不支持当前显卡或安装失败，本阶段停止并报告真实错误，不能
静默改回 CPU 训练。

## 3. 训练产物

所有新产物写入：

```text
results/dppo/gpu_smoke_v2/
```

目录分为：

- `dataset/`：当前 v2 状态和动作结构的极小仿真数据；
- `pretraining/`：GPU 扩散预训练检查点；
- `calibration/`：与新 v2 检查点绑定的稳定性 profile；
- `online/`：一次最小在线 DPPO 更新及训练历史。

旧的 `results/dppo/smoke_20260811/` 保留不动。它绑定 v1 schema，只作为历史
记录，不能被当前 v2 训练门禁复用。

## 4. 最小闭环规模

为了控制时间，只使用冒烟规模：

- 数据生成：6 个 Episode；
- 扩散预训练：2 个 epoch；
- 稳定性校准：沿用 debug 配置的每候选 1 次迭代、1 个 Episode；
- 在线训练：1 次迭代、1 个 Episode；
- 场景仍使用当前 5 个 MEC、3 个 VNF 和配置化 2–3 副本范围。

数据生成和 CLARABEL 属于仿真/数学求解，使用 CPU 是正常的；扩散预训练和在线
DPPO 网络更新必须使用 GPU。校准阶段也使用 GPU 重算策略概率和更新网络。

## 5. 验收标准

本阶段完成必须满足：

1. base 环境没有被修改或删除；
2. `rail-dppo-gpu` 能识别并使用 RTX 5060；
3. 新数据和检查点 metadata 使用 `dppo-v2-flat` 与
   `joint-sfc-continuous-v2`；
4. 稳定性 profile 与新预训练检查点的 SHA、配置 hash 和 v2 metadata 匹配；
5. 在线训练至少完成 1 次实际 optimizer update；
6. 在线闭环中的 DPPO 显式意图继续调用 CLARABEL 快层；
7. 生成 `dppo_online_best.pt`、`dppo_online_last.pt` 和
   `training_history.csv`；
8. 日志或检查点能够证明模型设备为 `cuda:0`；
9. 不覆盖旧结果，不运行基线或完整论文实验；
10. 若出现错误，只修复阻塞核心闭环的问题，并保留必要中文注释。

## 6. Git 范围

Conda 环境本身位于项目目录外，不提交到 Git。只有在真实运行暴露当前代码缺少
GPU 兼容处理时，才通过测试驱动修改项目文件并提交。训练结果默认不提交大体积
二进制检查点；代码、配置或文档变更继续上传到
`codex/dppo-main-algorithm` 分支。
