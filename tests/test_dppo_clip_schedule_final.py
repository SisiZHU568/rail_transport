import torch

from src.dppo_official_core import official_clip_schedule


def test_clip_schedule_uses_unique_geometric_formula() -> None:
    schedule = official_clip_schedule(
        fine_tuned_steps=3,
        maximum_clip_ratio=0.1,
        base_clip_ratio=0.001,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert torch.allclose(schedule, torch.tensor([0.001, 0.01, 0.1], dtype=torch.float64))
    assert official_clip_schedule(
        fine_tuned_steps=1,
        maximum_clip_ratio=0.1,
        base_clip_ratio=0.001,
        device=torch.device("cpu"),
        dtype=torch.float64,
    ).item() == 0.1
