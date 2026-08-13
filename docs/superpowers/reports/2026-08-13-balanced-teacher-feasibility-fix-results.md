# BalancedTeacher 可行性修复结果

## 修改内容

本次只修改 balanced 教师及其数据版本记录：

- balanced 副本数改为读取 `minimum_replicas`，默认由 3 个变为 2 个；
- balanced 节点选择改为可靠性优先，并继续让副本跨故障域分散；
- 保留时长公式、奖励函数、动作投影器和 CLARABEL 快层求解器均未修改；
- 新增 `teacher_schema_version: balanced-min-replica-reliability-v1`，使配置哈希自动隔离新旧专家数据和 checkpoint。

## 核心测试

运行教师、配置、数据集和教师诊断四组核心测试：

```text
51 passed in 2.30s
```

## 120 episode 数据门禁

使用 `configs/debug.yaml` 重新生成 120 episode、每个 episode 最多 2 个慢尺度步骤：

| 教师 | 候选数 | 接受数 | 拒绝数 | 接受率 |
|---|---:|---:|---:|---:|
| cost | 80 | 35 | 45 | 43.75% |
| reliability | 80 | 41 | 39 | 51.25% |
| balanced | 80 | 80 | 0 | 100.00% |

balanced 已从修改前的 2/80（2.5%）提高到 80/80（100%），超过设计门禁要求的 90%。

各分区有效标签数量：

| 分区 | cost | reliability | balanced | 合计 |
|---|---:|---:|---:|---:|
| train | 21 | 33 | 56 | 110 |
| validation | 8 | 4 | 10 | 22 |
| test | 6 | 4 | 14 | 24 |

三类教师均已覆盖训练集、验证集和测试集。

## Held-out 真实回放

对 validation 和 test 中的 46 条标签重新构建环境并执行：

| 教师 | 样本数 | 原始动作可行率 | 投影修改率 | 快层修复成功率 | 平均奖励 |
|---|---:|---:|---:|---:|---:|
| cost | 14 | 100% | 0% | 100% | -0.017106 |
| reliability | 8 | 100% | 0% | 100% | -0.011980 |
| balanced | 24 | 100% | 0% | 100% | -0.008923 |

balanced 的 CLARABEL 平均求解时间约为 6.78 ms，当前没有显示出快层耗时瓶颈。

## 投影字段说明

最初的教师连续提案需要通过约束投影，因此数据中的 `teacher_proposal_raw_feasible` 为否。数据收集器随后把投影后的离散意图重新编码为专家动作；真正用于行为克隆的 156 条保存标签满足：

- `expert_action_feasible`：156/156 为真；
- `final_feasible`：156/156 为真；
- validation/test 回放的投影修改率：0%；
- validation/test 回放的快层修复成功率：100%。

因此训练不会学习未修正的教师提案，也无需在奖励函数中加入投影修改惩罚。

## 结论与下一步

本次门禁通过，可以开始扩大专家数据规模。下一步应使用当前新配置哈希生成 1000 episode 数据，再重新执行专家预训练；旧教师规则生成的数据和 checkpoint 不应与新数据混用。
