# DPPO 稳定性校准实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将训练采样噪声、PPO 概率噪声和独立评估噪声彻底分离，加入全轨迹优势归一化与 KL 早停，并用可复现的短校准选择 `clip_ratio`，使正式在线训练只能使用通过校准且与当前配置、预训练检查点严格绑定的稳定性文件。

**Architecture:** 扩散调度器只计算理论反向标准差，DPPO 智能体在采样和概率计算两个使用点显式应用不同下限；稳定性配置由独立运行时配置模块统一解析，校准结果由独立领域模块验证、序列化和绑定；校准脚本复用正式训练的轨迹收集和 PPO 更新链路，但为每个候选值重新加载同一预训练策略；正式训练和评估均从在线检查点携带的稳定性元数据恢复，避免命令行参数漂移。

**Tech Stack:** Python 3.11、PyTorch、NumPy、PyYAML、pytest、CSV/JSON、现有 `DPPOAgent` 与慢时间尺度仿真环境。

---

## 实施边界

- 本计划只修改 DPPO 概率计算、PPO 更新稳定性、校准、检查点、训练入口和评估入口。
- 不修改奖励函数、状态空间、联合动作语义、约束投影、快尺度修复器和环境动力学。
- 所有新增关键公式、失败条件和配置字段必须带中文注释，注释解释“为什么这样做”，不能只复述代码。
- 每个任务都先写失败测试，再写最小实现；完成一个任务后立即运行目标测试并提交。
- 正式训练种子、校准种子和独立评估种子必须来自互不重叠的区间。
- 旧版在线检查点只作为历史冒烟产物保留；新代码明确拒绝缺少稳定性元数据的旧版在线检查点。

## Task 1：分离三类反向标准差下限

**Files:**

- Modify: `src/dppo_diffusion.py`
- Modify: `src/dppo.py`
- Modify: `tests/test_dppo_diffusion.py`
- Modify: `tests/test_dppo.py`

- [ ] **Step 1: 为调度器的显式标准差下限写失败测试**

在 `tests/test_dppo_diffusion.py` 增加测试，验证同一个理论反向标准差可由调用方分别限制到 `0.001`、`0.01` 和 `0.10`，并验证非正数下限会失败：

```python
def test_reverse_standard_deviation_uses_call_site_floor() -> None:
    schedule = CosineNoiseSchedule(steps=20)
    noisy_actions = torch.zeros((1, 3), dtype=torch.float32)
    final_timestep = torch.zeros(1, dtype=torch.long)

    evaluation_std = schedule.reverse_standard_deviation(
        noisy_actions,
        final_timestep,
        minimum_standard_deviation=0.001,
    )
    training_std = schedule.reverse_standard_deviation(
        noisy_actions,
        final_timestep,
        minimum_standard_deviation=0.01,
    )
    probability_std = schedule.reverse_standard_deviation(
        noisy_actions,
        final_timestep,
        minimum_standard_deviation=0.10,
    )

    assert torch.allclose(evaluation_std, torch.full_like(evaluation_std, 0.001))
    assert torch.allclose(training_std, torch.full_like(training_std, 0.01))
    assert torch.allclose(probability_std, torch.full_like(probability_std, 0.10))


@pytest.mark.parametrize("floor", [0.0, -0.1])
def test_reverse_standard_deviation_rejects_non_positive_floor(floor: float) -> None:
    schedule = CosineNoiseSchedule(steps=20)
    with pytest.raises(ValueError, match="minimum_standard_deviation"):
        schedule.reverse_standard_deviation(
            torch.zeros((1, 3)),
            torch.zeros(1, dtype=torch.long),
            minimum_standard_deviation=floor,
        )
```

- [ ] **Step 2: 运行目标测试并确认失败原因正确**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_diffusion.py -q
```

Expected: FAIL，提示 `reverse_standard_deviation()` 尚不接受 `minimum_standard_deviation`。

- [ ] **Step 3: 将标准差下限从调度器构造参数移到调用点**

在 `src/dppo_diffusion.py` 中删除 `CosineNoiseSchedule.__init__` 的 `minimum_reverse_standard_deviation` 参数和实例字段，将接口改为：

```python
def reverse_standard_deviation(
    self,
    noisy_actions: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    minimum_standard_deviation: float,
) -> torch.Tensor:
    """计算反向扩散标准差，并由当前使用场景显式指定数值下限。"""

    if minimum_standard_deviation <= 0.0:
        raise ValueError("minimum_standard_deviation 必须大于零。")
    variances = self._extract(
        self.posterior_variances,
        timesteps,
        noisy_actions,
    )
    minimum_variance = float(minimum_standard_deviation) ** 2
    return variances.clamp_min(minimum_variance).sqrt().expand_as(noisy_actions)
```

同步修改 `sample_reverse_diffusion(...)`，增加关键字参数 `minimum_sampling_standard_deviation: float = 0.001`，并显式传给调度器。预训练和通用采样继续使用 `0.001`，不改变既有预训练语义。

- [ ] **Step 4: 为智能体的采样/概率分离写失败测试**

在 `tests/test_dppo.py` 增加两组测试：

1. `sample_action` 返回的 `standard_deviations` 在最终去噪步不低于训练采样下限 `0.01`。
2. `sample_action` 保存旧策略 `log_probabilities` 时，对同一个实际采样动作改用概率下限 `0.10`，而不是复用采样下限 `0.01`。
3. `_current_trainable_log_probabilities` 重新计算新策略概率时同样使用概率下限 `0.10`。

测试配置必须显式构造：

```python
config = DPPOConfig(
    diffusion_steps=20,
    fine_tuned_steps=5,
    training_sampling_min_std=0.01,
    probability_min_std=0.10,
    evaluation_sampling_min_std=0.001,
)
```

概率测试使用 `unittest.mock.patch.object` 包装 `schedule.reverse_standard_deviation`，检查更新路径最后一次调用包含 `minimum_standard_deviation=0.10`。

- [ ] **Step 5: 扩展 `DPPOConfig` 并修改全部调用点**

在 `src/dppo.py` 中加入并校验：

```python
training_sampling_min_std: float = 0.01
probability_min_std: float = 0.10
evaluation_sampling_min_std: float = 0.001
```

`sample_action` 接口改为：

```python
def sample_action(
    self,
    states: np.ndarray | torch.Tensor,
    *,
    seed: int,
    sampling_min_std: float | None = None,
) -> DenoisingSample:
```

当 `sampling_min_std is None` 时使用 `self.config.training_sampling_min_std`。每个反向步骤先用采样标准差生成下一个实际动作，再针对同一均值和同一实际动作，用概率标准差计算并保存旧策略对数概率。`DenoisingSample.standard_deviations` 只表示实际采样标准差；`DenoisingSample.log_probabilities` 表示用概率下限 `0.10` 计算的旧策略概率。PPO 重新计算新策略概率时不复用采样标准差张量，而是调用：

```python
probability_std = self.schedule.reverse_standard_deviation(
    current_actions,
    timesteps,
    minimum_standard_deviation=self.config.probability_min_std,
)
```

更新 `src/dppo_diffusion.py`、`src/dppo.py` 以及测试中的全部旧调用点，保证每一处都显式说明用途。

- [ ] **Step 6: 运行目标测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_diffusion.py tests/test_dppo.py -q
```

Expected: PASS，零失败。

- [ ] **Step 7: 提交任务 1**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add src/dppo_diffusion.py src/dppo.py tests/test_dppo_diffusion.py tests/test_dppo.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "fix: separate DPPO diffusion noise floors"
```

## Task 2：加入优势归一化、稳定 KL 公式和更新早停

**Files:**

- Modify: `src/dppo.py`
- Modify: `tests/test_dppo.py`

- [ ] **Step 1: 为全轨迹优势归一化写失败测试**

在 `tests/test_dppo.py` 增加：

```python
def test_normalize_advantages_uses_whole_rollout_once() -> None:
    raw = np.asarray([1.0, 2.0, 5.0, 8.0], dtype=np.float32)
    normalized = normalize_advantages(raw)
    assert normalized.mean() == pytest.approx(0.0, abs=1e-6)
    assert normalized.std() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "raw",
    [
        np.asarray([3.0], dtype=np.float32),
        np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    ],
)
def test_normalize_advantages_keeps_degenerate_rollout(raw: np.ndarray) -> None:
    assert np.array_equal(normalize_advantages(raw), raw)
```

再通过固定 rollout 测试验证顺序为：先完成整个 rollout 的 GAE，再归一化一次，最后才按去噪步乘 `denoising_discount ** step_index`。不得归一化 reward、return 或已经展开后的去噪权重。

- [ ] **Step 2: 为 KL 公式和早停写失败测试**

增加纯函数公式测试：

```python
def test_approximate_kl_matches_non_negative_ratio_formula() -> None:
    old_logp = torch.tensor([-0.7, -0.2])
    new_logp = torch.tensor([-0.4, -0.8])
    log_ratio = new_logp - old_logp
    expected = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
    assert approximate_kl_divergence(new_logp, old_logp).item() == pytest.approx(
        expected.item()
    )
```

再构造极小 `target_kl` 的智能体，断言 `update()` 返回：

```python
assert metrics["kl_early_stopped"] == 1.0
assert metrics["optimizer_step_count"] < config.update_epochs * expected_batches
assert metrics["maximum_approximate_kl"] >= config.target_kl
```

- [ ] **Step 3: 运行测试并确认缺少函数/指标**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo.py -q
```

Expected: FAIL，原因是归一化函数、KL 函数、早停字段尚不存在。

- [ ] **Step 4: 实现优势归一化和 KL 计算**

在 `src/dppo.py` 增加：

```python
def normalize_advantages(
    advantages: np.ndarray,
    *,
    minimum_standard_deviation: float = 1e-8,
) -> np.ndarray:
    """在展开去噪步骤前，对整段环境轨迹的 GAE 优势只归一化一次。"""

    values = np.asarray(advantages, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("advantages 必须是一维数组。")
    if values.size < 2:
        return values.copy()
    standard_deviation = float(values.std())
    if standard_deviation <= minimum_standard_deviation:
        return values.copy()
    return ((values - float(values.mean())) / standard_deviation).astype(
        np.float32,
        copy=False,
    )


def approximate_kl_divergence(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
) -> torch.Tensor:
    """使用 PPO 常用的非负近似式，避免直接均值抵消正负偏移。"""

    log_ratio = new_log_probabilities - old_log_probabilities
    return torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
```

在 `DPPOAgent.update()` 中，紧接 `compute_gae(...)` 之后只对环境级 GAE 数组调用一次 `normalize_advantages`，然后才转换为 `outer_advantages`，并在小批量阶段按可训练去噪步骤应用 `denoising_discount`。

- [ ] **Step 5: 实现 KL 阈值早停**

向 `DPPOConfig` 增加：

```python
target_kl: float = 1.0
normalize_advantages: bool = True
```

在 `DPPOAgent.update()` 的每个小批量中，先计算当前新旧策略 KL 和裁剪比例并记录诊断值；当 `current_kl >= target_kl` 时，不执行本小批量及剩余小批量的反向传播。`optimizer_step_count` 只统计已经完成的策略优化器步骤；即使第一个批次就早停，损失均值和梯度范数也必须返回定义清楚的有限默认值，不能对空列表求均值。返回指标至少包含：

```python
{
    "approximate_kl": mean_approximate_kl,
    "maximum_approximate_kl": maximum_approximate_kl,
    "clip_fraction": mean_clip_fraction,
    "optimizer_step_count": float(optimizer_step_count),
    "kl_early_stopped": float(kl_early_stopped),
}
```

中文注释必须说明：KL 已达到阈值时继续优化只会让新策略离旧策略更远，所以当前批次不再执行优化器步骤。早停属于正常完成，更新结束后仍清空 rollout buffer。

- [ ] **Step 6: 运行 DPPO 智能体测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo.py -q
```

Expected: PASS，零失败。

- [ ] **Step 7: 提交任务 2**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add src/dppo.py tests/test_dppo.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "fix: stabilize DPPO PPO updates"
```

## Task 3：建立统一稳定性配置和校准文件模型

**Files:**

- Create: `src/dppo_training_config.py`
- Create: `src/dppo_stability.py`
- Create: `tests/test_dppo_stability.py`
- Modify: `configs/debug.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 为 YAML 配置解析写失败测试**

在 `tests/test_config.py` 验证 `configs/debug.yaml` 包含以下结构和类型：

```yaml
dppo:
  stability:
    training_sampling_min_std: 0.01
    probability_min_std: 0.10
    evaluation_sampling_min_std: 0.001
    target_kl: 1.0
    target_clip_fraction_min: 0.10
    target_clip_fraction_max: 0.20
    clip_ratio_candidates: [0.10, 0.01, 0.001]
    calibration_iterations: 3
    calibration_episodes_per_iteration: 2
    calibration_seed_start: 20000
```

校准种子从 `20000` 开始，正式训练从现有 `13000` 开始，独立评估由命令行显式指定且测试必须使用 `30000` 之后的范围。

- [ ] **Step 2: 为候选资格和选择规则写失败测试**

在 `tests/test_dppo_stability.py` 覆盖：

- 全部指标有限、`maximum_approximate_kl < 1.0`、平均裁剪比例位于闭区间 `[0.10, 0.20]` 且至少执行一次优化器步骤时，候选合格。
- `0.10` 和 `0.01` 同时合格时选择数值更大的 `0.10`，不按输入顺序选择。
- KL 达阈值、裁剪比例越界、零优化步骤或非有限值分别产生清晰失败原因。
- 无候选合格时仍生成 `qualified=False` 的文件，`selected_clip_ratio=None`。
- 配置哈希或预训练检查点 SHA256 不匹配时验证失败。
- JSON 保存后重新加载，所有不可变字段完全一致，文件摘要可复现。

- [ ] **Step 3: 运行测试并确认新模块尚不存在**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_stability.py tests/test_config.py -q
```

Expected: FAIL，提示 `src.dppo_stability` 或新配置字段不存在。

- [ ] **Step 4: 实现统一运行时配置解析**

在 `src/dppo_training_config.py` 定义：

```python
@dataclass(frozen=True)
class DPPOStabilitySettings:
    training_sampling_min_std: float
    probability_min_std: float
    evaluation_sampling_min_std: float
    target_kl: float
    target_clip_fraction_min: float
    target_clip_fraction_max: float
    clip_ratio_candidates: tuple[float, ...]
    calibration_iterations: int
    calibration_episodes_per_iteration: int
    calibration_seed_start: int


```

公共解析接口固定为 `load_dppo_stability_settings(config: Mapping[str, Any]) -> DPPOStabilitySettings`；智能体配置接口固定为 `build_dppo_agent_config(config: Mapping[str, Any], *, clip_ratio: float) -> DPPOConfig`。

实现时完整校验正数、目标裁剪区间顺序、候选非空且互不重复、迭代数/episode 数为正整数。`build_dppo_agent_config` 是训练和校准唯一的 YAML 到 `DPPOConfig` 转换入口，避免两个脚本生成不同智能体参数。

- [ ] **Step 5: 实现稳定性领域模型**

在 `src/dppo_stability.py` 定义不可变数据类：

```python
STABILITY_PROFILE_SCHEMA_VERSION = "dppo-stability-v1"

@dataclass(frozen=True)
class DPPOCalibrationCandidateResult:
    clip_ratio: float
    mean_clip_fraction: float
    mean_approximate_kl: float
    maximum_approximate_kl: float
    optimizer_step_count: int
    finite: bool
    qualified: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DPPOStabilityProfile:
    schema_version: str
    qualified: bool
    selected_clip_ratio: float | None
    config_hash: str
    pretrained_checkpoint_sha256: str
    training_sampling_min_std: float
    probability_min_std: float
    evaluation_sampling_min_std: float
    target_kl: float
    target_clip_fraction_min: float
    target_clip_fraction_max: float
    candidate_values: tuple[float, ...]
    episode_seeds: tuple[int, ...]
    iterations_per_candidate: int
    episodes_per_iteration: int
    candidate_results: tuple[DPPOCalibrationCandidateResult, ...]
    failure_reasons: tuple[str, ...]
```

同时实现以下公共接口：

- `sha256_file(path: str | Path) -> str`
- `select_stability_profile(*, config_hash: str, pretrained_checkpoint_sha256: str, settings: DPPOStabilitySettings, episode_seeds: Sequence[int], candidate_results: Sequence[DPPOCalibrationCandidateResult]) -> DPPOStabilityProfile`
- `stability_profile_json(profile: DPPOStabilityProfile) -> str`，生成不依赖文件换行风格的规范化 JSON 字符串。
- `parse_stability_profile_json(serialized: str) -> DPPOStabilityProfile`
- `save_stability_profile(path: str | Path, profile: DPPOStabilityProfile) -> None`
- `load_stability_profile(path: str | Path) -> DPPOStabilityProfile`
- `stability_profile_sha256(profile: DPPOStabilityProfile) -> str`
- `validate_stability_profile(profile: DPPOStabilityProfile, *, config_hash: str, pretrained_checkpoint_path: str | Path, settings: DPPOStabilitySettings) -> None`

JSON 使用稳定键顺序、UTF-8 和固定缩进；摘要对同一规范化 JSON 字节计算。验证函数首先拒绝 `qualified=False`，随后检查配置哈希、检查点摘要以及三个标准差下限、KL 阈值和裁剪目标区间。

- [ ] **Step 6: 修改 debug 配置并运行测试**

在 `configs/debug.yaml` 加入上述 `dppo.stability` 节，使用中文注释解释三类标准差的不同职责和种子隔离要求。

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_stability.py tests/test_config.py -q
```

Expected: PASS，零失败。

- [ ] **Step 7: 提交任务 3**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add src/dppo_training_config.py src/dppo_stability.py tests/test_dppo_stability.py configs/debug.yaml tests/test_config.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "feat: add DPPO stability profile model"
```

## Task 4：实现公平、可复现的短校准命令

**Files:**

- Create: `run_dppo_stability_calibration.py`
- Create: `tests/test_dppo_stability_calibration.py`
- Modify: `run_dppo_training.py`
- Modify: `tests/test_dppo_training.py`

- [ ] **Step 1: 将训练公共函数改为可复用公共接口**

在 `run_dppo_training.py` 中保留 `collect_rollout(...)`，删除私有 `_agent_config(...)`，所有构建智能体配置的代码改用 `src.dppo_training_config.build_dppo_agent_config(...)`。`_checkpoint_metadata(...)` 暂时保持原位，校准脚本用相同字段构造预训练兼容性元数据。

- [ ] **Step 2: 为校准公平性写失败测试**

在 `tests/test_dppo_stability_calibration.py` 使用小型测试环境和临时预训练检查点，验证：

- 三个候选收到完全相同的 episode seed 序列。
- 每个候选开始时策略、价值网络和优化器均来自全新加载，不继承前一候选更新结果。
- 候选 CSV 按候选值降序写一行，包含资格判断使用的全部指标和失败原因。
- `stability_profile.json` 无论是否有候选合格都必须生成。
- 两次使用相同配置、检查点和种子运行时产生相同选择结果。

为便于测试，脚本公开接口 `calibrate_dppo_stability(*, config: dict[str, Any], pretrained_checkpoint: str | Path, output_root: str | Path, device: str) -> DPPOStabilityProfile`。

- [ ] **Step 3: 运行测试并确认入口缺失**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_stability_calibration.py -q
```

Expected: FAIL，提示 `run_dppo_stability_calibration` 不存在。

- [ ] **Step 4: 实现候选运行和聚合**

新脚本按以下固定流程运行每个候选：

1. 重新构建环境。
2. 用 `build_dppo_agent_config(config, clip_ratio=candidate)` 构建全新智能体。
3. 从同一个预训练检查点加载策略权重。
4. 按 `calibration_seed_start + local_episode_index` 收集完全相同的轨迹种子序列。
5. 每个外层迭代调用一次 `agent.update()`。
6. 聚合所有更新的 `clip_fraction`、`approximate_kl`、`maximum_approximate_kl` 和 `optimizer_step_count`。
7. 调用 `select_stability_profile(...)`，不在脚本内复制资格公式。

关键循环采用明确的重新加载边界：

```python
for clip_ratio in sorted(settings.clip_ratio_candidates, reverse=True):
    environment = build_dppo_environment(config)
    agent = _load_fresh_pretrained_agent(
        config=config,
        environment=environment,
        clip_ratio=clip_ratio,
        pretrained_checkpoint=pretrained_checkpoint,
        device=device,
    )
    candidate_results.append(
        _run_candidate(
            environment=environment,
            agent=agent,
            episode_seeds=episode_seeds,
            iterations=settings.calibration_iterations,
            episodes_per_iteration=settings.calibration_episodes_per_iteration,
        )
    )
```

中文注释解释为何每个候选必须全新加载，以及为什么三组候选必须共用同一组种子。

- [ ] **Step 5: 实现 CLI 和两个产物**

CLI 参数固定为：

```text
--config
--pretrained-checkpoint   必填
--output-root             必填
--device                  cpu 或 cuda
```

输出：

- `candidate_metrics.csv`
- `stability_profile.json`

若无候选合格，命令仍正常写出两个诊断产物，然后以非零退出码结束；不得悄悄回退到 YAML 原有 `clip_ratio`。

- [ ] **Step 6: 运行校准测试和训练回归测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_stability_calibration.py tests/test_dppo_training.py -q
```

Expected: PASS，零失败。

- [ ] **Step 7: 提交任务 4**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add run_dppo_stability_calibration.py run_dppo_training.py tests/test_dppo_stability_calibration.py tests/test_dppo_training.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "feat: add reproducible DPPO stability calibration"
```

## Task 5：升级在线检查点并绑定稳定性文件

**Files:**

- Modify: `src/dppo_checkpoint.py`
- Modify: `tests/test_dppo_checkpoint.py`

- [ ] **Step 1: 为在线检查点 v2 写失败测试**

在 `tests/test_dppo_checkpoint.py` 验证：

- 保存在线检查点时必须传入已通过校准的 `DPPOStabilityProfile`。
- 加载后 `LoadedDPPOOnlineCheckpoint.stability_profile` 与原对象完全一致。
- 载荷同时保存 `stability_profile_sha256`，加载时重新计算并检查。
- 篡改稳定性字段或摘要会拒绝加载。
- 格式版本为 `dppo-online-checkpoint-v2`。
- v1 在线检查点抛出明确错误，说明它缺少正式训练所需的稳定性绑定。

- [ ] **Step 2: 运行检查点测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_checkpoint.py -q
```

Expected: FAIL，提示保存接口或加载结果中没有稳定性字段。

- [ ] **Step 3: 修改保存和加载接口**

将保存接口改为：

```python
def save_dppo_online_checkpoint(
    path: str | Path,
    agent: DPPOAgent,
    metadata: DPPOCheckpointMetadata,
    *,
    iteration: int,
    best_mean_reward: float,
    stability_profile: DPPOStabilityProfile,
) -> None:
```

将加载结果改为：

```python
@dataclass(frozen=True)
class LoadedDPPOOnlineCheckpoint:
    agent: DPPOAgent
    metadata: DPPOCheckpointMetadata
    iteration: int
    best_mean_reward: float
    rng_state: torch.Tensor
    stability_profile: DPPOStabilityProfile
    stability_profile_sha256: str
```

检查点载荷使用明确字段保存：

```python
"stability_profile_json": stability_profile_json(stability_profile),
"stability_profile_sha256": stability_profile_sha256(stability_profile),
```

保存前拒绝 `qualified=False` 或 `selected_clip_ratio is None`；加载时用 `parse_stability_profile_json(...)` 恢复对象，并验证 schema、摘要、`profile.config_hash == metadata.config_hash`、`agent.config.clip_ratio == profile.selected_clip_ratio`，以及智能体配置中的三类标准差和 KL 阈值均与 profile 一致，再返回对象。

- [ ] **Step 4: 运行检查点测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_checkpoint.py -q
```

Expected: PASS，零失败。

- [ ] **Step 5: 提交任务 5**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add src/dppo_checkpoint.py tests/test_dppo_checkpoint.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "feat: bind DPPO checkpoints to stability profiles"
```

## Task 6：正式训练强制消费合格校准文件

**Files:**

- Modify: `run_dppo_training.py`
- Modify: `tests/test_dppo_training.py`

- [ ] **Step 1: 为训练入口的强制验证写失败测试**

在 `tests/test_dppo_training.py` 增加：

- `parse_arguments` 缺少 `--stability-profile` 时退出。
- 不合格文件拒绝训练。
- 配置哈希不匹配拒绝训练。
- 预训练检查点 SHA256 不匹配拒绝训练。
- 合格文件覆盖 YAML 中的原始 `clip_ratio`，训练智能体使用 `selected_clip_ratio`。
- best/last 两个检查点均携带同一稳定性文件及摘要。
- 训练历史新增稳定性字段和 KL 早停字段。

- [ ] **Step 2: 运行训练测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_training.py -q
```

Expected: FAIL，提示 CLI、训练签名或历史列尚未更新。

- [ ] **Step 3: 更新 CLI 和正式训练启动顺序**

增加必填参数：

```python
parser.add_argument(
    "--stability-profile",
    required=True,
    help="校准命令生成且通过验证的 stability_profile.json。",
)
```

`main()` 必须按以下顺序执行：

1. 加载 YAML 和稳定性设置。
2. 加载稳定性文件。
3. 计算当前配置哈希和预训练检查点 SHA256。
4. 调用 `validate_stability_profile(...)`。
5. 用 `profile.selected_clip_ratio` 构建智能体配置。
6. 加载预训练权重。
7. 开始收集第一条正式训练轨迹。

验证失败时不得创建训练输出目录或写入部分检查点。

- [ ] **Step 4: 更新训练历史与检查点调用**

在 `TRAINING_HISTORY_COLUMNS` 增加：

```python
"maximum_approximate_kl",
"optimizer_step_count",
"kl_early_stopped",
"selected_clip_ratio",
"stability_profile_sha256",
```

由于摘要是字符串，将训练历史行类型从 `dict[str, float]` 调整为 `dict[str, float | str]`。`train_dppo(...)` 明确接收：

```python
def train_dppo(
    environment: DPPOSlowTimescaleEnvironment,
    agent: DPPOAgent,
    metadata: DPPOCheckpointMetadata,
    *,
    iterations: int,
    episodes_per_iteration: int,
    seed_start: int,
    output_root: str | Path,
    stability_profile: DPPOStabilityProfile,
) -> tuple[dict[str, float | str], ...]:
```

保存 best/last 检查点时均传入同一 `stability_profile`。打印日志增加当前最大 KL、裁剪比例和是否提前停止。

- [ ] **Step 5: 运行训练与检查点测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_training.py tests/test_dppo_checkpoint.py -q
```

Expected: PASS，零失败。

- [ ] **Step 6: 提交任务 6**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add run_dppo_training.py tests/test_dppo_training.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "feat: require calibrated DPPO training profiles"
```

## Task 7：评估使用独立噪声并输出稳定性元数据

**Files:**

- Modify: `src/dppo_evaluation.py`
- Modify: `run_dppo_evaluation.py`
- Modify: `tests/test_dppo_evaluation.py`

- [ ] **Step 1: 为评估噪声来源写失败测试**

在 `tests/test_dppo_evaluation.py` 增加测试，用 mock 验证 `evaluate_dppo_episode(...)` 调用：

```python
agent.sample_action(
    state,
    seed=expected_seed,
    sampling_min_std=profile.evaluation_sampling_min_std,
)
```

另加报告测试，验证 `write_evaluation_reports(...)` 生成 `stability_metadata.json`，至少包含以下字段：

```json
{
  "schema_version": "dppo-stability-v1",
  "selected_clip_ratio": 0.1,
  "training_sampling_min_std": 0.01,
  "probability_min_std": 0.1,
  "evaluation_sampling_min_std": 0.001,
  "target_kl": 1.0
}
```

并断言 `stability_profile_sha256` 等于测试中对规范化稳定性文件重新计算得到的 64 位十六进制摘要。

- [ ] **Step 2: 运行评估测试并确认失败**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_evaluation.py -q
```

Expected: FAIL，提示评估函数没有标准差参数或缺少元数据产物。

- [ ] **Step 3: 将评估标准差沿调用链显式传递**

修改 `src/dppo_evaluation.py`。`evaluate_dppo_episode` 的完整签名为 `evaluate_dppo_episode(environment: DPPOSlowTimescaleEnvironment, agent: DPPOAgent, *, episode_seed: int, evaluation_sampling_min_std: float) -> DPPOEpisodeEvaluation`；批量接口为 `evaluate_dppo_episodes(environment: DPPOSlowTimescaleEnvironment, agent: DPPOAgent, *, episodes: int, seed_start: int, evaluation_sampling_min_std: float) -> tuple[DPPOEpisodeEvaluation, ...]`。

在动作采样处把该值传给 `sample_action`。中文注释说明评估噪声只控制评估探索幅度，不参与 PPO 概率计算。

- [ ] **Step 4: 从在线检查点读取并报告稳定性元数据**

`run_dppo_evaluation.py` 不新增可覆盖噪声的命令行参数，直接使用：

```python
profile = loaded.stability_profile
evaluations = evaluate_dppo_episodes(
    environment,
    loaded.agent,
    episodes=parsed.episodes,
    seed_start=parsed.seed_start,
    evaluation_sampling_min_std=profile.evaluation_sampling_min_std,
)
```

扩展 `write_evaluation_reports(...)`：

```python
def write_evaluation_reports(
    evaluations: Sequence[DPPOEpisodeEvaluation],
    output_root: str | Path,
    *,
    stability_profile: DPPOStabilityProfile,
) -> dict[str, Path]:
```

新增 `stability_metadata.json`，并在返回路径字典中加入 `"stability_metadata"`。

- [ ] **Step 5: 运行评估测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_evaluation.py -q
```

Expected: PASS，零失败。

- [ ] **Step 6: 提交任务 7**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add src/dppo_evaluation.py run_dppo_evaluation.py tests/test_dppo_evaluation.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "feat: evaluate DPPO with calibrated metadata"
```

## Task 8：端到端冒烟验证、文档和范围审计

**Files:**

- Modify: `README.md`
- Modify: `tests/test_active_algorithm_scope.py`
- Modify: `tests/test_dppo_training.py`
- Modify: `tests/test_dppo_evaluation.py`

- [ ] **Step 1: 增加端到端失败测试**

在测试临时目录内执行最小链路：

1. 使用现有辅助函数创建微型预训练检查点。
2. 校准三个候选并生成 profile。
3. 使用 profile 运行一次正式训练迭代。
4. 加载最后在线检查点。
5. 用 `30000` 起始的独立种子评估一个 episode。
6. 断言三个阶段的种子集合互不相交。
7. 断言 profile 摘要在训练历史、在线检查点和评估元数据中一致。

测试不得依赖仓库外已有的模型文件或结果目录。

- [ ] **Step 2: 扩展活动算法范围审计**

在 `tests/test_active_algorithm_scope.py` 中把以下文件加入 DPPO 主算法允许列表：

```text
src/dppo_training_config.py
src/dppo_stability.py
run_dppo_stability_calibration.py
```

继续断言活动源码、配置和 README 中不存在 DDQN 主算法入口；历史设计文档不参与活动源码扫描。

- [ ] **Step 3: 更新 README 操作顺序**

README 明确写成四步：

```powershell
# 1. 生成教师仿真数据并完成扩散预训练
# 2. 对固定预训练检查点执行短校准
D:\Anaconda3\python.exe run_dppo_stability_calibration.py --config configs/debug.yaml --pretrained-checkpoint results/dppo/pretraining/dppo_pretrained.pt --output-root results/dppo/calibration --device cpu

# 3. 只使用合格稳定性文件进行正式在线训练
D:\Anaconda3\python.exe run_dppo_training.py --config configs/debug.yaml --pretrained-checkpoint results/dppo/pretraining/dppo_pretrained.pt --stability-profile results/dppo/calibration/stability_profile.json --output-root results/dppo/training --device cpu

# 4. 使用与训练种子隔离的新种子独立评估
D:\Anaconda3\python.exe run_dppo_evaluation.py --config configs/debug.yaml --checkpoint results/dppo/training/last.pt --episodes 20 --seed-start 30000 --output-root results/dppo/evaluation --device cpu
```

说明 `results/dppo/pretraining/dppo_pretrained.pt` 是现有预训练命令生成的准确文件名；校准只是稳定性门禁，不是论文最终性能实验。

- [ ] **Step 4: 运行 DPPO 专项测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest tests/test_dppo_diffusion.py tests/test_dppo.py tests/test_dppo_stability.py tests/test_dppo_stability_calibration.py tests/test_dppo_checkpoint.py tests/test_dppo_training.py tests/test_dppo_evaluation.py tests/test_active_algorithm_scope.py -q
```

Expected: PASS，零失败。

- [ ] **Step 5: 运行完整回归测试**

Run:

```powershell
D:\Anaconda3\python.exe -m pytest -q
```

Expected: 全部测试通过，零失败、零错误。

- [ ] **Step 6: 检查变更范围和占位内容**

Run:

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' diff --check
Get-ChildItem src,tests -Recurse -File | Select-String -Pattern 'TODO|TBD|pass\s*$|NotImplementedError'
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' status --short
```

Expected:

- `git diff --check` 无输出。
- 新增/修改的 DPPO 文件没有占位实现。
- 工作区只包含本任务预期文件。

- [ ] **Step 7: 人工 review 清单**

逐项确认：

- 三个标准差下限分别只用于训练采样、PPO 概率和独立评估。
- 优势只在环境 GAE 完成后、去噪折扣展开前归一化一次。
- KL 使用 `mean((exp(log_ratio) - 1) - log_ratio)`。
- 达到 KL 阈值的当前批次没有执行优化器步骤。
- 每个候选从同一预训练状态独立开始并共享相同种子。
- 选择的是数值最大的合格候选，不是第一个候选。
- 无合格候选时正式训练无法启动。
- 配置和预训练检查点任一变化都会使旧 profile 失效。
- 在线检查点、训练历史、评估报告携带一致的 profile 摘要。
- 奖励、状态、动作、投影和快层修复代码未发生语义变化。
- 新增关键逻辑含适合初学者 review 的中文注释。

- [ ] **Step 8: 提交最终文档与集成测试**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' add README.md tests/test_active_algorithm_scope.py tests/test_dppo_training.py tests/test_dppo_evaluation.py
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' commit -m "docs: document calibrated DPPO workflow"
```

- [ ] **Step 9: 推送现有分支并更新 PR**

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe' push origin codex/dppo-main-algorithm
```

Expected: 推送成功，现有 PR 自动包含全部新提交。
