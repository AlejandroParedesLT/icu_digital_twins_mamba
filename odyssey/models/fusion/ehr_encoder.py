"""Stay-level EHR encoder wrappers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from odyssey.models.fusion.utils import masked_mean_pool


class StayEHRMambaEncoder(nn.Module):
    """Wrap an EHR Mamba-style model and return stay-level embeddings."""

    def __init__(
        self,
        pretrained_model: Optional[nn.Module] = None,
        hidden_size: int = 768,
        pooling: str = "mean",
        vocab_size: int = 32000,
    ) -> None:
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden_size = hidden_size
        self.pooling = pooling
        if self.pretrained_model is None:
            self.fallback_embedding = nn.Embedding(vocab_size, hidden_size)
            self.fallback_norm = nn.LayerNorm(hidden_size)

    def _compute_hidden_states(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract token-level hidden states from the wrapped EHR model."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )

        if self.pretrained_model is None:
            embeddings = self.fallback_embedding(batch["concept_ids"])
            return self.fallback_norm(embeddings)

        if hasattr(self.pretrained_model, "embeddings") and hasattr(self.pretrained_model, "model"):
            inputs_embeds = self.pretrained_model.embeddings(
                input_ids=batch["concept_ids"],
                token_type_ids_batch=batch["type_ids"],
                time_stamps=batch["time_stamps"],
                ages=batch["ages"],
                visit_orders=batch["visit_orders"],
                visit_segments=batch["visit_segments"],
            )
            backbone = getattr(self.pretrained_model.model, "backbone", None)
            if backbone is not None:
                outputs = backbone(inputs_embeds=inputs_embeds)
                return outputs.last_hidden_state

        if hasattr(self.pretrained_model, "encode"):
            hidden_states = self.pretrained_model.encode(inputs)
            if hidden_states.dim() == 2:
                return hidden_states.unsqueeze(1)
            return hidden_states

        outputs = self.pretrained_model(
            inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        if getattr(outputs, "hidden_states", None):
            return outputs.hidden_states[-1]
        if getattr(outputs, "last_hidden_state", None) is not None:
            return outputs.last_hidden_state
        raise ValueError("Wrapped EHR model does not expose usable hidden states.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode a stay-level EHR batch."""
        token_states = self._compute_hidden_states(batch)
        attention_mask = batch.get("attention_mask")

        if self.pooling == "cls":
            pooled = token_states[:, 0]
        else:
            pooled = masked_mean_pool(token_states, attention_mask)

        return {
            "sequence": token_states,
            "pooled": pooled,
        }
