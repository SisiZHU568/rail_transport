# DPPO 副本数量上下限配置化设计

## 目标

移除 DPPO 动作中写死的“2/3 副本”语义。每个 VNF 仍只使用一个连续
冗余动作分量，但副本数量允许在配置给出的闭区间内独立选择。扩大 MEC、
故障域或可靠性实验规模时，只修改配置，不修改 Python 源码。

本任务只修改副本数量编码及其兼容性边界，不实现新的快时间尺度数学求解器。

## 配置模式

`dppo.action` 使用以下唯一配置：

```yaml
minimum_replicas: 2
maximum_replicas: 3
```

两个值必须是正整数，且满足：

```text
1 <= minimum_replicas <= maximum_replicas <= compute_node_count
```

现有 `replica_threshold` 删除，不保留兼容别名，避免同一动作同时存在阈值模式
和区间量化模式。debug 配置暂时使用 2 到 3，正式实验可改为 2 到 5 或其他
合法范围。

## 连续动作解码

设副本候选数量为：

```text
K = maximum_replicas - minimum_replicas + 1
```

每个 VNF 的连续冗余分数 `s` 先截断到 `[-1, 1]`，再均匀映射到 `K` 个区间：

```text
index = min(floor(((s + 1) / 2) * K), K - 1)
replica_count = minimum_replicas + index
```

当上下限相等时，所有分数都解码为该固定副本数。动作维度保持不变，仍为每个
VNF 一个冗余分数，因此提高最大副本数不会扩大扩散网络输出维度。

## 教师动作编码

教师给出的整数副本数必须位于配置区间。编码时使用对应量化区间的中心值：

```text
score = -1 + 2 * (index + 0.5) / K
```

这样教师动作经过“编码—解码”后能够严格恢复原副本数量，且不会落在区间边界。

三类教师按配置解释副本数：

- 成本教师使用 `minimum_replicas`；
- 可靠性教师使用 `maximum_replicas`；
- 平衡教师使用 `min(minimum_replicas + 1, maximum_replicas)`。

## 投影和执行边界

投影器不再只接受 2 或 3，而是接受动作空间已经解码出的合法正整数。它仍须
拒绝超过可用计算节点数量、节点重复、资源不足或故障域不足的动作。

快层继续严格执行每个 VNF 的解码副本数，不能擅自增减。新的数学求解器将在
后续独立任务中复用这一配置化结果。

## 检查点兼容性

动作模式版本从 `joint-sfc-continuous-v1` 升级为
`joint-sfc-continuous-v2`。检查点元数据删除 `replica_threshold`，新增：

- `minimum_replicas`；
- `maximum_replicas`。

旧预训练和在线检查点必须明确拒绝加载并重新生成，因为相同连续分数在两个版本
中可能代表不同副本数量。不得静默转换旧检查点。

## 代码范围

主要修改：

- `configs/debug.yaml`；
- `src/config.py`；
- `src/dppo_action_space.py`；
- `src/dppo_teacher.py`；
- `src/dppo_projection.py`；
- `src/dppo_scenario.py`；
- `src/dppo_checkpoint.py`；
- 构造检查点元数据的 DPPO 运行脚本；
- 直接覆盖上述行为的核心测试。

不修改 DPPO 网络结构、状态维度、动作维度、奖励函数和当前快层算法。

## 核心测试

只运行直接相关的核心测试，覆盖：

1. 配置拒绝布尔值、非正数、上下限倒置和超过计算节点数；
2. `[2, 5]` 的四个副本数都可独立解码；
3. `minimum_replicas == maximum_replicas` 的固定范围；
4. 教师动作在任意合法区间内编码—解码一致；
5. 三类教师使用配置化副本数；
6. 投影器接受 3 以上的合法副本数并拒绝超过节点数；
7. 检查点绑定上下限并拒绝旧动作模式或不匹配配置。

## 完成标准

- 活动 DPPO 代码不再读取或保存 `replica_threshold`；
- 活动 DPPO 代码不再用 `(2, 3)` 校验副本数；
- 每个 VNF 仍只有一个连续冗余动作维度；
- debug 配置和新检查点模式可以完成核心测试；
- 代码包含必要中文注释，并明确提示旧检查点需要重新生成。
