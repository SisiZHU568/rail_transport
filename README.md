# Rail Serverless SFC 双时间尺度编排

## 官方 DPPO 核心适配

本项目的慢时间尺度主算法使用
[irom-princeton/dppo](https://github.com/irom-princeton/dppo) 的官方 DPPO
策略损失与去噪 MDP 设计，并适配到轨道交通 Serverless SFC 联合编排环境。
上游许可证和来源说明保存在 `third_party/irom_dppo/`。

### 当前重构进度

阶段 A 已建立配置驱动的部署边界、精确 CTMC 故障快照、实例批次生命周期和实例成本账本。阶段 B 已建立跨时隙队列、不可变在途与完成事件、确定性 EDF 以及批次级 SLA/排空审计。阶段 C 使用两阶段 CLARABEL 连续凸优化联合分配无线、计算和有线商品流。阶段 D–E 已把配置驱动的观察/动作规格、安全部署解码、真实双时间尺度环境和事务式在线 PPO 更新接入主链。

阶段 A 核心检查使用已创建的 `rail-dppo-gpu` 环境：

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_orchestration_config.py tests/test_failure_process.py tests/test_instance_lifecycle.py tests/test_cost_ledger.py tests/test_phase_a_orchestration.py -q
```

阶段 B 核心检查：

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_queue_state.py tests/test_queue_manager.py tests/test_sla_drain.py tests/test_phase_b_orchestration.py -q
```

接口约定如下：

- `To = 1`：一个慢时间尺度状态作为当前观测；
- `Ta = 1`：每个慢时隙一次生成整套联合编排动作；
- `Do`：由状态编码器根据场景配置计算；
- `Da`：由联合动作空间根据 MEC、VNF 和 SFC 规模计算。

因此扩大 MEC 或 VNF 数量时，不需要修改 DPPO 网络源码，只需更新配置和场景。
轨道环境继续负责动作解码、约束投影和快时间尺度修复，未引入官方仓库中的
MuJoCo、IsaacGym、WandB 或机器人任务依赖。

运行官方核心和轨道训练闭环的精简测试：

```powershell
python -m pytest tests/test_dppo_official_core.py tests/test_dppo.py tests/test_phase_e_environment.py tests/test_phase_e_online_trainer.py -q
```

### 最小完整训练闭环

正式在线训练只接受 `phase-e-pretrained-v1` checkpoint 和带观察/动作规格哈希的 Phase E 教师数据。旧 `dppo-checkpoint-v1`、`dppo-online-checkpoint-v2` 及旧字段数据集不会自动迁移。

使用通过安全解码复验的教师数据进行预训练：

```powershell
python run_phase_e_pretraining.py --dataset results/phase_e/teachers.npz --output results/phase_e/pretrained.pt --observation-spec-hash <observation-sha256> --action-spec-hash <action-sha256> --steps 200 --device cuda
```

运行 32 个真实慢帧并完成两个事务 PPO 更新：

```powershell
python run_phase_e_online_training.py --config configs/debug.yaml --pretrained-checkpoint results/phase_e/pretrained.pt --teacher-dataset results/phase_e/teachers.npz --output results/phase_e/online --frames 32 --device cuda
```

从完整更新边界恢复到更大的总帧数：

```powershell
python run_phase_e_online_training.py --config configs/debug.yaml --pretrained-checkpoint results/phase_e/pretrained.pt --teacher-dataset results/phase_e/teachers.npz --output results/phase_e/online --frames 64 --resume-checkpoint results/phase_e/online/dppo_phase_e_last.pt --device cuda
```

成功后输出目录包含 `dppo_phase_e_best.pt`、`dppo_phase_e_last.pt` 和 `training_history.csv`。内部求解失败时写入 `failure_audit.json`，当前未提交 rollout 不进入 PPO。

### 双时间尺度主算法

- 慢层 DPPO 一次生成整条 SFC 的副本数量、部署节点和保留时间；
- 快层 CVXPY + CLARABEL 在已部署温实例间联合分配上行、计算和有线商品流；
- 两阶段词典序目标先最小化紧迫度加权服务缺口，再最小化资源成本；
- 求解结果必须通过有限性、物理残差和四版本提交审计，不调用旧枚举器或启发式回退。

DPPO 神经网络可在 CUDA 上更新；CLARABEL 数学求解始终使用 CPU。快层配置位于
`fast_resource_optimization`。每 16 个完整慢帧才提交一次 PPO 更新，内部失败会
丢弃当前事务 rollout。正式训练 checkpoint 同时保存策略、价值网络、两个优化器、
更新计数和随机数状态。

检查本机求解环境：

```powershell
python -c "import cvxpy as cp; print(cp.__version__, cp.installed_solvers())"
```
