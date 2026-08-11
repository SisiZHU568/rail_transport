# DPPO upstream notice

This project adapts the DPPO algorithm core from:

- Repository: <https://github.com/irom-princeton/dppo>
- Copyright: 2024 Intelligent Robot Motion Lab
- License: MIT License (see `LICENSE` in this directory)

The adapted implementation mainly follows:

- `model/diffusion/diffusion_ppo.py`
- `model/diffusion/diffusion_vpg.py`
- `agent/finetune/train_ppo_diffusion_agent.py`

The tensor interface and training loop were adapted for the rail Serverless SFC
environment. Robotics environments, rendering, experiment tracking and other
upstream application-specific components are not copied into this repository.
