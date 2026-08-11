# Rail Serverless SFC 双时间尺度编排

## 官方 DPPO 核心适配

本项目的慢时间尺度主算法使用
[irom-princeton/dppo](https://github.com/irom-princeton/dppo) 的官方 DPPO
策略损失与去噪 MDP 设计，并适配到轨道交通 Serverless SFC 联合编排环境。
上游许可证和来源说明保存在 `third_party/irom_dppo/`。

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
python -m pytest tests/test_dppo_official_core.py tests/test_dppo.py tests/test_config.py tests/test_dppo_training.py -q
```

### 最小完整训练闭环

下面的命令只用于验证整条链路，不代表论文正式训练规模：

```powershell
python run_dppo_dataset_generation.py --config configs/debug.yaml --episodes 6 --seed-start 41000 --output-root results/dppo/smoke/dataset

python run_dppo_pretraining.py --config configs/debug.yaml --dataset-root results/dppo/smoke/dataset --output-root results/dppo/smoke/pretraining --epochs 2 --device cpu

python run_dppo_stability_calibration.py --config configs/debug.yaml --pretrained-checkpoint results/dppo/smoke/pretraining/dppo_pretrained.pt --output-root results/dppo/smoke/calibration --device cpu

python run_dppo_training.py --config configs/debug.yaml --iterations 2 --episodes-per-iteration 1 --pretrained-checkpoint results/dppo/smoke/pretraining/dppo_pretrained.pt --stability-profile results/dppo/smoke/calibration/stability_profile.json --output-root results/dppo/smoke/online --device cpu
```

成功后，`online` 目录应包含 `dppo_online_best.pt`、
`dppo_online_last.pt` 和 `training_history.csv`。debug 配置只用于快速验证；
论文实验应使用独立配置扩大数据量、校准 Episode 和在线训练迭代数。
