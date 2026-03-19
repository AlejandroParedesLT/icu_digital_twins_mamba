"""Cross-attention multimodal stay-level fusion."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from odyssey.models.fusion.cde_encoder import StayNCDEEncoder
from odyssey.models.fusion.ehr_encoder import StayEHRMambaEncoder
from odyssey.models.fusion.image_encoder import StayImageEncoder
from odyssey.models.fusion.utils import MODALITY_ORDER, ProjectionHead, build_modality_mask


class CrossAttentionBlock(nn.Module):
    """One round of cross-attention followed by a feed-forward update."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention from fusion queries into one modality context."""
        attended, _ = self.cross_attention(queries, context, context)
        queries = self.norm(queries + attended)
        return queries + self.feed_forward(queries)


class CrossAttentionFusionModel(nn.Module):
    """Fuse EHR, CDE, and image modalities with learned fusion tokens."""

    def __init__(
        self,
        ehr_encoder: StayEHRMambaEncoder,
        cde_encoder: StayNCDEEncoder,
        image_encoder: StayImageEncoder,
        ehr_dim: int = 768,
        cde_dim: int = 32,
        image_dim: int = 768,
        fusion_dim: int = 256,
        num_fusion_tokens: int = 4,
        num_layers: int = 2,
        num_heads: int = 8,
        num_tasks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ehr_encoder = ehr_encoder
        self.cde_encoder = cde_encoder
        self.image_encoder = image_encoder
        self.num_tasks = num_tasks

        self.ehr_projection = ProjectionHead(ehr_dim, fusion_dim, dropout)
        self.cde_projection = ProjectionHead(cde_dim, fusion_dim, dropout)
        self.image_projection = ProjectionHead(image_dim, fusion_dim, dropout)
        self.modality_embeddings = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(1, 1, fusion_dim))
                for modality in MODALITY_ORDER
            }
        )
        self.fusion_tokens = nn.Parameter(torch.randn(1, num_fusion_tokens, fusion_dim))
        self.fusion_layers = nn.ModuleList(
            [CrossAttentionBlock(fusion_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_tasks),
        )

    def _project_sequence(
        self,
        sequence: torch.Tensor,
        projection: ProjectionHead,
        modality: str,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Project a modality sequence into the fusion dimension."""
        projected = projection(sequence)
        projected = projected + self.modality_embeddings[modality]
        return projected * mask.unsqueeze(-1)

    def forward(
        self,
        ehr_batch: Dict[str, torch.Tensor],
        cde_coeffs: torch.Tensor,
        images: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse all modalities with learned cross-attention tokens."""
        device = ehr_batch["concept_ids"].device
        batch_size = ehr_batch["concept_ids"].shape[0]
        mask = build_modality_mask(batch_size, device, modality_mask)

        ehr_outputs = self.ehr_encoder(ehr_batch)
        cde_outputs = self.cde_encoder(cde_coeffs)
        image_outputs = self.image_encoder(images)

        ehr_sequence = self._project_sequence(
            ehr_outputs["sequence"],
            self.ehr_projection,
            "ehr",
            mask["ehr"],
        )
        cde_sequence = self._project_sequence(
            cde_outputs["sequence"],
            self.cde_projection,
            "cde",
            mask["cde"],
        )
        img_sequence = self._project_sequence(
            image_outputs["sequence"],
            self.image_projection,
            "img",
            mask["img"],
        )

        fusion = self.fusion_tokens.expand(batch_size, -1, -1)
        for layer in self.fusion_layers:
            fusion = layer(fusion, ehr_sequence)
            fusion = layer(fusion, cde_sequence)
            fusion = layer(fusion, img_sequence)

        pooled = fusion.mean(dim=1)
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "fused": pooled,
            "fusion_tokens": fusion,
            "ehr_sequence": ehr_sequence,
            "cde_sequence": cde_sequence,
            "img_sequence": img_sequence,
            "mask_vector": torch.cat([mask[modality] for modality in MODALITY_ORDER], dim=1),
        }
