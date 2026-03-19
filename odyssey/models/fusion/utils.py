"""Helpers for stay-level multimodal fusion."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


MODALITY_ORDER = ("ehr", "cde", "img")


def build_modality_mask(
    batch_size: int,
    device: torch.device,
    overrides: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Build a default-present modality mask dictionary."""
    mask = {
        modality: torch.ones(batch_size, 1, device=device, dtype=torch.float32)
        for modality in MODALITY_ORDER
    }
    if overrides:
        for modality, values in overrides.items():
            mask[modality] = values.to(device=device, dtype=torch.float32).view(batch_size, 1)
    return mask


def masked_mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool a token sequence with an optional binary mask."""
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    weights = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom


class ProjectionHead(nn.Module):
    """Project modality outputs into a common fusion dimension."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project modality embeddings to the shared fusion dimension."""
        return self.net(inputs)
