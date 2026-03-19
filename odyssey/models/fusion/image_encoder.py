"""Stay-level image encoder wrappers."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


class StayImageEncoder(nn.Module):
    """Encode one or more images per stay into pooled embeddings."""

    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        hidden_size: int = 768,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size

        if self.base_model is None:
            self.fallback = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_size),
            )

    def _encode_single_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of individual images."""
        if self.base_model is not None:
            outputs = self.base_model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)
            return outputs

        return self.fallback(images)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a stay-level image batch.

        Accepts either `(batch, channels, height, width)` or
        `(batch, num_images, channels, height, width)`.
        """
        if images.dim() == 4:
            per_image = self._encode_single_images(images).unsqueeze(1)
        elif images.dim() == 5:
            batch_size, num_images, channels, height, width = images.shape
            flat_images = images.view(batch_size * num_images, channels, height, width)
            per_image = self._encode_single_images(flat_images).view(batch_size, num_images, -1)
        else:
            raise ValueError("Images must have rank 4 or 5.")

        pooled = per_image.mean(dim=1)
        return {
            "sequence": per_image,
            "pooled": pooled,
        }
