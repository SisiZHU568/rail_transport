# Phase C Lexicographic Convex Fast Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox syntax.

**Goal:** 用快照驱动的两阶段 CLARABEL 连续凸优化替换旧路径比例快层，联合分配上行带宽/功率、温实例计算时间和有线流量。

**Architecture:** 新 `FastResourceOptimizer` 是无状态纯求解器，读取队列、生命周期、故障、网络和只读配置，先最小化紧迫度加权服务缺口，再固定一级近最优值最小化货币成本。输出阶段 B 已定义的 `FastAllocationPlan`，实际状态变更仍由队列提交端完成。

**Tech Stack:** Python 3.12、CVXPY、CLARABEL、NumPy、pytest。

---

### Task 1: 快层配置和网络快照

- Create `src/fast_resource_model.py`, `tests/test_fast_resource_model.py`。
- 测试先行定义无线、节点、VNF、链路和数值尺度的严格不可变配置。
- 配置加入 `configs/debug.yaml` 并由 `src/config.py` 启动校验。
- 运行专项测试，提交 `feat: define fast resource snapshots`。

### Task 2: 单节点无线与计算两阶段凸优化

- Create `src/fast_resource_optimizer.py`, `tests/test_fast_resource_optimizer.py`。
- RED 覆盖 DCP、空队列、零信道、温实例限制、有限服务缺口、一级优先和二级成本。
- GREEN 使用 `rel_entr` 无线透视、`PowCone3D` 计算能耗和连续两次 CLARABEL。
- 审计状态、耗时、词典序容差、有限性和残差，提交 `feat: add lexicographic fast resource optimizer`。

### Task 3: 有线目标商品流与计划提交

- 扩展求解器和测试：正容量链路、距离/跳数下降势、绑定目标不可改指、多商品共享容量、传播时隙。
- 输出阶段 B `AllocationOperation`，提交端校验四版本；失败返回 `FAST_SOLVER_FAILURE` 且无回退。
- 提交 `feat: add causal wired commodity allocation`。

### Task 4: 接入并删除旧快层语义

- 修改 `src/dppo_scenario.py`、`src/fast_slot_executor.py` 和相关配置/测试使用新快层。
- 删除 `src/fast_convex_scheduler.py`、旧路径比例测试及冷实例回退入口；不保留兼容开关。
- 运行阶段 A–C 核心回归、静态检查并推送。
