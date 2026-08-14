# BalancedTeacher Feasibility Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 balanced 教师标签可行率过低的问题，并用新的 120 episode 数据验证扩大训练规模的前置门禁。

**Architecture:** 保留现有三教师结构，仅让 `BalancedTeacher` 使用配置下限副本数和已有可靠性排序，再继续调用故障域分散逻辑。新增非空教师规则版本字段，使现有配置哈希自动隔离新旧数据及 checkpoint。

**Tech Stack:** Python 3.12、PyTorch、pytest、YAML、现有 DPPO 仿真环境与 CLARABEL 快层求解器。

---

### Task 1: 锁定 balanced 新行为

**Files:**
- Modify: `tests/test_dppo_teacher.py`
- Modify: `src/dppo_teacher.py`

- [ ] 将默认 balanced 期望副本数由 3 改为 2。
- [ ] 将可配置副本边界测试中的 balanced 期望值改为配置下限。
- [ ] 增加 balanced 前缀与可靠性排序一致、且故障域不同的行为断言。
- [ ] 运行教师测试并确认旧实现因副本数和排序不符而失败。
- [ ] 最小修改 `BalancedTeacher._replica_count` 和 `_rank_nodes`，补充必要中文注释。
- [ ] 再次运行教师测试并确认通过。

### Task 2: 隔离新旧教师数据

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/config.py`
- Modify: `configs/debug.yaml`

- [ ] 增加配置包含固定教师规则版本的断言。
- [ ] 增加版本缺失、空字符串和非字符串均被拒绝的测试。
- [ ] 运行配置测试并确认失败原因来自缺少校验或字段。
- [ ] 在 YAML 中加入 `balanced-min-replica-reliability-v1`，在配置校验中要求非空字符串。
- [ ] 再次运行配置测试并确认通过。

### Task 3: 核心回归与 120 episode 门禁

**Files:**
- Create: `results/dppo/datasets/teacher_fix_120_v1/*`（运行产物，不提交 Git）
- Create: `results/dppo/diagnostics/teacher_fix_120_v1/*`（运行产物，不提交 Git）
- Modify: `docs/superpowers/reports/2026-08-13-balanced-teacher-feasibility-fix-results.md`

- [ ] 运行教师、配置、数据集和教师诊断核心测试。
- [ ] 以 120 episodes 和新输出目录生成专家数据。
- [ ] 对 validation/test 标签运行真实回放诊断。
- [ ] 核对 balanced 总体候选通过率至少 90%，且三类教师均覆盖 train/validation/test。
- [ ] 记录实际计数、通过率和下一阶段结论。
- [ ] 运行 `git diff --check`，提交并推送代码与报告。
