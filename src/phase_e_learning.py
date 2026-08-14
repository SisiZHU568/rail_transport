"""阶段 E 的轻量辅助学习与在线行为克隆调度。"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AuxiliaryLossResult:
    rank_loss: float
    count_loss: float
    retention_loss: float
    total_loss: float
    tensor: torch.Tensor


class PhaseEAuxiliaryHeads(nn.Module):
    """只读取观察编码，不读取教师动作、扩散噪声或带噪动作。"""

    def __init__(self, state_dim: int, pair_count: int, count_classes: int,
                 retention_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.SiLU())
        self.rank = nn.Linear(hidden_dim, pair_count)
        self.count = nn.Linear(hidden_dim, pair_count * count_classes)
        self.retention = nn.Linear(hidden_dim, pair_count * retention_classes)
        self.pair_count = pair_count
        self.count_classes = count_classes
        self.retention_classes = retention_classes

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.encoder(observations)
        return (
            self.rank(encoded),
            self.count(encoded).reshape(-1, self.pair_count, self.count_classes),
            self.retention(encoded).reshape(-1, self.pair_count, self.retention_classes),
        )


def auxiliary_teacher_loss(
    *,
    rank_scores: torch.Tensor,
    count_logits: torch.Tensor,
    retention_logits: torch.Tensor,
    selected_nodes: torch.Tensor,
    count_targets: torch.Tensor,
    retention_targets: torch.Tensor,
    retention_mask: torch.Tensor,
    beta_aux: float = 0.1,
) -> AuxiliaryLossResult:
    """三个任务等权；只有无语义的保留时间维度被屏蔽。"""
    selected_nodes = selected_nodes.bool()
    retention_mask = retention_mask.bool()
    selected = rank_scores[selected_nodes]
    unselected = rank_scores[~selected_nodes]
    if selected.numel() and unselected.numel():
        rank_loss = F.softplus(-(selected[:, None] - unselected[None, :])).mean()
    else:
        rank_loss = rank_scores.sum() * 0.0
    count_loss = F.cross_entropy(
        count_logits.reshape(-1, count_logits.shape[-1]), count_targets.reshape(-1)
    )
    if retention_mask.any():
        retention_loss = F.cross_entropy(
            retention_logits[retention_mask], retention_targets[retention_mask]
        )
    else:
        retention_loss = retention_logits.sum() * 0.0
    total = beta_aux * (rank_loss + count_loss + retention_loss) / 3.0
    return AuxiliaryLossResult(
        float(rank_loss.detach()), float(count_loss.detach()),
        float(retention_loss.detach()), float(total.detach()), total
    )


def online_bc_weight(update_index: int, *, total_updates: int) -> float:
    """在线前 20% 更新从 0.05 线性衰减到零。"""
    if total_updates <= 0 or update_index < 0:
        raise ValueError("更新编号和总更新数无效。")
    decay_updates = math.ceil(0.2 * total_updates)
    return 0.05 * max(0.0, 1.0 - update_index / decay_updates)
