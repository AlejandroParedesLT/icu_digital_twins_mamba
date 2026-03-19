"""Late-fusion multimodal stay-level models."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from odyssey.models.fusion.cde_encoder import StayNCDEEncoder
from odyssey.models.fusion.ehr_encoder import StayEHRMambaEncoder
from odyssey.models.fusion.image_encoder import StayImageEncoder
from odyssey.models.fusion.utils import MODALITY_ORDER, ProjectionHead, build_modality_mask


class LateFusionModel(nn.Module):
    """Concatenate modality embeddings and classify from the fused representation."""

    def __init__(
        self,
        ehr_encoder: StayEHRMambaEncoder,
        cde_encoder: StayNCDEEncoder,
        image_encoder: StayImageEncoder,
        ehr_dim: int = 768,
        cde_dim: int = 32,
        image_dim: int = 768,
        fusion_dim: int = 256,
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

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3 + len(MODALITY_ORDER)),
            nn.Linear(fusion_dim * 3 + len(MODALITY_ORDER), fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, num_tasks),
        )

    def encode_modalities(
        self,
        ehr_batch: Dict[str, torch.Tensor],
        cde_coeffs: torch.Tensor,
        images: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode all modalities and project them into a shared space."""
        device = ehr_batch["concept_ids"].device
        batch_size = ehr_batch["concept_ids"].shape[0]
        mask = build_modality_mask(batch_size, device, modality_mask)

        ehr_outputs = self.ehr_encoder(ehr_batch)
        cde_outputs = self.cde_encoder(cde_coeffs)
        image_outputs = self.image_encoder(images)

        ehr = self.ehr_projection(ehr_outputs["pooled"]) * mask["ehr"]
        cde = self.cde_projection(cde_outputs["pooled"]) * mask["cde"]
        img = self.image_projection(image_outputs["pooled"]) * mask["img"]

        mask_vector = torch.cat([mask[modality] for modality in MODALITY_ORDER], dim=1)

        return {
            "ehr": ehr,
            "cde": cde,
            "img": img,
            "mask_vector": mask_vector,
            "ehr_sequence": ehr_outputs["sequence"],
            "cde_sequence": cde_outputs["sequence"],
            "img_sequence": image_outputs["sequence"],
        }

    def forward(
        self,
        ehr_batch: Dict[str, torch.Tensor],
        cde_coeffs: torch.Tensor,
        images: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run late fusion over the three modality encoders."""
        encoded = self.encode_modalities(ehr_batch, cde_coeffs, images, modality_mask)
        fused = torch.cat(
            [
                encoded["ehr"],
                encoded["cde"],
                encoded["img"],
                encoded["mask_vector"],
            ],
            dim=1,
        )
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "fused": fused,
            **encoded,
        }


class GatedFusionModel(LateFusionModel):
    """Late fusion with learned per-modality gates."""

    def __init__(
        self,
        ehr_encoder: StayEHRMambaEncoder,
        cde_encoder: StayNCDEEncoder,
        image_encoder: StayImageEncoder,
        ehr_dim: int = 768,
        cde_dim: int = 32,
        image_dim: int = 768,
        fusion_dim: int = 256,
        num_tasks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            ehr_encoder=ehr_encoder,
            cde_encoder=cde_encoder,
            image_encoder=image_encoder,
            ehr_dim=ehr_dim,
            cde_dim=cde_dim,
            image_dim=image_dim,
            fusion_dim=fusion_dim,
            num_tasks=num_tasks,
            dropout=dropout,
        )
        self.ehr_gate = nn.Linear(fusion_dim, fusion_dim)
        self.cde_gate = nn.Linear(fusion_dim, fusion_dim)
        self.img_gate = nn.Linear(fusion_dim, fusion_dim)

    def forward(
        self,
        ehr_batch: Dict[str, torch.Tensor],
        cde_coeffs: torch.Tensor,
        images: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run gated late fusion over the three modality encoders."""
        encoded = self.encode_modalities(ehr_batch, cde_coeffs, images, modality_mask)

        gated_ehr = torch.sigmoid(self.ehr_gate(encoded["ehr"])) * encoded["ehr"]
        gated_cde = torch.sigmoid(self.cde_gate(encoded["cde"])) * encoded["cde"]
        gated_img = torch.sigmoid(self.img_gate(encoded["img"])) * encoded["img"]

        fused = torch.cat(
            [
                gated_ehr,
                gated_cde,
                gated_img,
                encoded["mask_vector"],
            ],
            dim=1,
        )
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "fused": fused,
            "ehr": gated_ehr,
            "cde": gated_cde,
            "img": gated_img,
            "mask_vector": encoded["mask_vector"],
            "ehr_sequence": encoded["ehr_sequence"],
            "cde_sequence": encoded["cde_sequence"],
            "img_sequence": encoded["img_sequence"],
        }
