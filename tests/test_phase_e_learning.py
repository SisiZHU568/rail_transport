import pytest
import torch

from src.phase_e_learning import auxiliary_teacher_loss, online_bc_weight


def test_auxiliary_loss_masks_only_retention_targets() -> None:
    rank_scores = torch.tensor([[2.0, -1.0]], requires_grad=True)
    count_logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], requires_grad=True)
    retention_logits = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]], requires_grad=True)
    result = auxiliary_teacher_loss(
        rank_scores=rank_scores,
        count_logits=count_logits,
        retention_logits=retention_logits,
        selected_nodes=torch.tensor([[True, False]]),
        count_targets=torch.tensor([[1, 0]]),
        retention_targets=torch.tensor([[1, 0]]),
        retention_mask=torch.tensor([[True, False]]),
    )

    assert result.rank_loss > 0
    assert result.count_loss > 0  # 零实例仍是有效分类目标。
    assert result.retention_loss == pytest.approx(
        torch.nn.functional.cross_entropy(retention_logits[0, :1], torch.tensor([1])).item()
    )
    assert result.total_loss == pytest.approx(
        0.1 * (result.rank_loss + result.count_loss + result.retention_loss) / 3
    )


def test_online_bc_is_used_only_during_first_twenty_percent_updates() -> None:
    assert online_bc_weight(0, total_updates=10) == pytest.approx(0.05)
    assert online_bc_weight(1, total_updates=10) == pytest.approx(0.025)
    assert online_bc_weight(2, total_updates=10) == 0.0
    assert online_bc_weight(9, total_updates=10) == 0.0
