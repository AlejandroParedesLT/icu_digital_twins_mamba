"""Stay-level NCDE encoder wrappers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class StayNCDEEncoder(nn.Module):
    """Wrap an NCDE model and expose stay-level latent embeddings."""

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        input_channels: int = 19,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.input_channels = input_channels
        self.hidden_size = hidden_size

        if self.base_model is None:
            self.fallback = nn.Sequential(
                nn.LazyLinear(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )

    def forward(self, coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode CDE coefficients into stay-level embeddings."""
        if self.base_model is not None:
            if hasattr(self.base_model, "encode"):
                pooled = self.base_model.encode(coeffs)
            else:
                outputs = self.base_model(coeffs)
                if isinstance(outputs, tuple):
                    pooled = outputs[-1]
                else:
                    pooled = outputs
        else:
            pooled = self.fallback(coeffs.flatten(start_dim=1))

        if pooled.dim() > 2:
            pooled = pooled[:, -1]

        return {
            "sequence": pooled.unsqueeze(1),
            "pooled": pooled,
        }
