# DPPO GPU Smoke Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建隔离的 `rail-dppo-gpu` Conda 环境，并完成当前 v2 双时间尺度主算法的一次可复现 GPU 冒烟训练闭环。

**Architecture:** 新环境只承载 CUDA PyTorch 和当前项目所需依赖，代码仍从现有工作树运行。四阶段产物写入新的 `results/dppo/gpu_smoke_v2`；DPPO 网络在 GPU 上训练，CVXPY + CLARABEL 在 CPU 上求解快层。任何 GPU、schema、稳定性门禁或快层错误都显式停止，不回退 CPU 训练。

**Tech Stack:** Conda、Python 3.12、PyTorch CUDA、NumPy 1.26、CVXPY 1.6、CLARABEL、PyYAML、pytest

---

## Task 1：创建并验证隔离 GPU 环境

**Environment:** `rail-dppo-gpu`

- [ ] **Step 1：确认目标环境尚不存在**

```powershell
D:\Anaconda3\Scripts\conda.exe env list
```

Expected: 没有 `rail-dppo-gpu`；若已存在，先检查而不删除，避免覆盖未知环境。

- [ ] **Step 2：创建干净的 Python 3.12 环境**

```powershell
D:\Anaconda3\Scripts\conda.exe create -n rail-dppo-gpu python=3.12 pip -y
```

Expected: 命令成功，且 `D:\Anaconda3\envs\rail-dppo-gpu\python.exe` 存在。

- [ ] **Step 3：安装官方 CUDA PyTorch**

优先使用官方 CUDA 12.8 wheel 索引；RTX 5060 需要较新的 CUDA 架构支持：

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Expected: 安装的 `torch.__version__` 不带 `+cpu`，`torch.version.cuda` 非空。

- [ ] **Step 4：安装项目最小依赖**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pip install "numpy==1.26.4" "cvxpy==1.6.7" "PyYAML>=6,<7" "pytest>=8,<9"
```

Expected: 安装成功。不要安装项目未使用的训练框架或基线依赖。

- [ ] **Step 5：验证 GPU、CLARABEL 和依赖一致性**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "import cvxpy as cp, numpy as np, torch, yaml; assert torch.cuda.is_available(); x=torch.tensor([1.0,2.0], device='cuda:0'); print('torch',torch.__version__); print('cuda',torch.version.cuda); print('gpu',torch.cuda.get_device_name(0)); print('tensor_device',x.device); print('numpy',np.__version__); print('solvers',cp.installed_solvers()); assert 'CLARABEL' in cp.installed_solvers()"
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pip check
```

Expected:

- GPU 为 `NVIDIA GeForce RTX 5060`；
- tensor device 为 `cuda:0`；
- solver 列表包含 `CLARABEL`；
- `pip check` 输出 `No broken requirements found.`。

若 `torch.cuda.is_available()` 为 False，立即停止，不执行后续 CPU 替代流程。

## Task 2：在新环境运行代码和快层核心预检

**Files:**

- Read: `configs/debug.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_fast_convex_scheduler.py`
- Test: `tests/test_dppo_slow_timescale_env.py`

- [ ] **Step 1：验证配置、CLARABEL 和真实慢窗口**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pytest tests/test_config.py tests/test_fast_convex_scheduler.py tests/test_dppo_slow_timescale_env.py -q --basetemp="C:\Users\Administrator\.codex\.chatgpt-projects\g-p-6a6ff4453abc819184ee35997a150260\.pytest_tmp\gpu_env_core"
```

Expected: 全部通过。测试会证明新环境能同时加载 CUDA PyTorch 和 CPU CLARABEL。

- [ ] **Step 2：确认当前 schema 和输出目录**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "from src.config import load_config; c=load_config('configs/debug.yaml'); print(c['dppo']['training']['state_schema_version']); print(c['dppo']['action']['schema_version']); print(c['dppo']['fast_scheduler'])"
```

Expected:

```text
dppo-v2-flat
joint-sfc-continuous-v2
{'solver': 'CLARABEL', ...}
```

## Task 3：生成当前 v2 极小仿真数据

**Output:** `results/dppo/gpu_smoke_v2/dataset`

- [ ] **Step 1：检查输出目标不存在**

```powershell
Test-Path results\dppo\gpu_smoke_v2
```

Expected: `False`。若已存在，不删除也不覆盖；改用带时间后缀的新目录并在最终报告中注明。

- [ ] **Step 2：生成 6 个 Episode**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_dataset_generation.py --config configs/debug.yaml --episodes 6 --seed-start 42000 --output-root results/dppo/gpu_smoke_v2/dataset
```

Expected: 生成 `train.npz`、`validation.npz`、`test.npz`、`diagnostics.npz` 和 `metadata.json`。

- [ ] **Step 3：校验 v2 数据 metadata**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "import json; from pathlib import Path; m=json.loads(Path('results/dppo/gpu_smoke_v2/dataset/metadata.json').read_text(encoding='utf-8')); print(m); assert m['state_schema_version']=='dppo-v2-flat'; assert m['action_schema_version']=='joint-sfc-continuous-v2'"
```

Expected: 两个断言通过，数据维度来自当前 5 MEC、3 VNF 配置。

## Task 4：在 GPU 上完成极小扩散预训练

**Output:** `results/dppo/gpu_smoke_v2/pretraining`

- [ ] **Step 1：运行 2 个 epoch 的 GPU 预训练**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_pretraining.py --config configs/debug.yaml --dataset-root results/dppo/gpu_smoke_v2/dataset --output-root results/dppo/gpu_smoke_v2/pretraining --epochs 2 --device cuda
```

Expected: 生成 `dppo_pretrained.pt`，日志显示设备为 CUDA，损失为有限值。

- [ ] **Step 2：校验检查点 schema 和 GPU RNG 状态**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "import torch; p=torch.load('results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt', map_location='cpu', weights_only=False); m=p['metadata']; print(m); assert m['state_schema_version']=='dppo-v2-flat'; assert m['action_schema_version']=='joint-sfc-continuous-v2'; assert p['cuda_rng_states'] is not None"
```

Expected: v2 schema 断言通过，并保存 CUDA 随机状态以便复现。

## Task 5：使用 GPU 重新生成稳定性 profile

**Output:** `results/dppo/gpu_smoke_v2/calibration`

- [ ] **Step 1：执行稳定性校准**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_stability_calibration.py --config configs/debug.yaml --pretrained-checkpoint results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt --output-root results/dppo/gpu_smoke_v2/calibration --device cuda
```

Expected: 生成 `candidate_metrics.csv` 和 `stability_profile.json`。至少一个候选必须 `qualified=true`；否则停止，不绕过门禁。

- [ ] **Step 2：校验 profile 与新检查点绑定**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "from pathlib import Path; from src.config import load_config; from src.dppo_stability import load_stability_profile, validate_stability_profile; p=load_stability_profile(Path('results/dppo/gpu_smoke_v2/calibration/stability_profile.json')); validate_stability_profile(profile=p, config=load_config('configs/debug.yaml'), pretrained_checkpoint_path=Path('results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt')); print(p.qualified, p.selected_clip_ratio); assert p.qualified"
```

Expected: 校验通过并输出 `True` 和选定 clip ratio。

## Task 6：运行一次 GPU 在线 DPPO 更新

**Output:** `results/dppo/gpu_smoke_v2/online`

- [ ] **Step 1：执行 1 iteration × 1 Episode 在线训练**

```powershell
$env:PYTHONUTF8='1'
D:\Anaconda3\envs\rail-dppo-gpu\python.exe run_dppo_training.py --config configs/debug.yaml --iterations 1 --episodes-per-iteration 1 --pretrained-checkpoint results/dppo/gpu_smoke_v2/pretraining/dppo_pretrained.pt --stability-profile results/dppo/gpu_smoke_v2/calibration/stability_profile.json --output-root results/dppo/gpu_smoke_v2/online --device cuda
```

Expected:

- 完成至少一次 optimizer update；
- 训练指标均为有限值；
- 在线窗口中的 DPPO 显式意图使用 CLARABEL 快层；
- 生成 `dppo_online_best.pt`、`dppo_online_last.pt`、`training_history.csv`。

- [ ] **Step 2：检查训练历史和 v2 在线检查点**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "import csv, math, torch; from pathlib import Path; rows=list(csv.DictReader(Path('results/dppo/gpu_smoke_v2/online/training_history.csv').open(encoding='utf-8'))); print(rows); assert len(rows)==1; assert all(math.isfinite(float(rows[0][k])) for k in ('mean_episode_reward','policy_loss','value_loss')); p=torch.load('results/dppo/gpu_smoke_v2/online/dppo_online_last.pt', map_location='cpu', weights_only=False); assert p['metadata']['state_schema_version']=='dppo-v2-flat'; assert p['metadata']['action_schema_version']=='joint-sfc-continuous-v2'; assert p['cuda_rng_states'] is not None"
```

Expected: 一行有限训练历史，在线检查点为 v2 且保存 CUDA RNG 状态。

## Task 7：失败修复边界和最终报告

**Files:**

- Modify only if a real blocker is found: the exact affected `src/*.py` and matching `tests/test_*.py`
- Do not commit: `results/dppo/gpu_smoke_v2/**/*.pt`

- [ ] **Step 1：若遇到代码缺陷，执行最小 TDD 修复**

每个真实缺陷必须：

1. 在对应现有测试文件写一个最小失败测试；
2. 运行该测试取得 RED；
3. 只修改阻塞 GPU/v2/CLARABEL 闭环的代码；
4. 运行目标测试取得 GREEN；
5. 提交并上传 `codex/dppo-main-algorithm`。

不得因为训练慢、指标不理想或显存利用率不高而增加算法模块。

- [ ] **Step 2：最终环境与产物验证**

```powershell
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -m pip check
D:\Anaconda3\envs\rail-dppo-gpu\python.exe -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.memory_allocated(0)); assert torch.cuda.is_available()"
Get-ChildItem results\dppo\gpu_smoke_v2 -Recurse -File | Select-Object FullName,Length
D:\Git\cmd\git.exe status --short
```

Expected: 环境无依赖冲突，GPU 仍可用，四阶段产物齐全；Git 不包含检查点或意外代码变更。

- [ ] **Step 3：向用户报告可 review 的结果**

报告必须包含：

- 新 Conda 环境路径和激活方式；
- PyTorch、CUDA、GPU、CVXPY、CLARABEL 版本；
- 四阶段实际运行命令与耗时；
- 新产物路径和各文件作用；
- 数据/检查点 schema；
- 训练历史关键指标；
- 是否发生代码修复及对应 commit；
- 当前只是主算法冒烟闭环，不代表正式实验性能。
