"""Mamba model."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import (
    MambaCausalLMOutput,
    MambaForCausalLM,
)

from odyssey.models.ehr_mamba.mamba_utils import (
    MambaForMultiHeadSequenceClassification,
    MambaForSequenceClassification,
    MambaSequenceClassifierOutput,
)
from odyssey.models.embeddings import MambaEmbeddingsForCEHR
# from odyssey.models.

class MambaPretrain(pl.LightningModule):
    """Mamba model for pretraining."""

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 768,
        time_embeddings_size: int = 32,
        visit_order_size: int = 3,
        type_vocab_size: int = 9,
        max_num_visits: int = 512,
        max_seq_length: int = 2048,
        state_size: int = 16,
        num_hidden_layers: int = 32,
        expand: int = 2,
        conv_kernel: int = 4,
        learning_rate: float = 5e-5,
        dropout_prob: float = 0.1,
        padding_idx: int = 0,
        cls_idx: int = 5,
        use_mambapy: bool = False,
        # Image-specific parameters
        image_size: int = 224,
        patch_size: int = 16,
        image_encoder_dim: int = 768,
        image_encoder_depth: int = 6,
        image_encoder_heads: int = 8,
        fusion_method: str = "add",  # Options: "add", "concat", "mlp"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.visit_order_size = visit_order_size
        self.type_vocab_size = type_vocab_size
        self.max_num_visits = max_num_visits
        self.max_seq_length = max_seq_length
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.expand = expand
        self.conv_kernel = conv_kernel
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.padding_idx = padding_idx
        self.cls_idx = cls_idx
        self.use_mambapy = use_mambapy
        
         # Image-specific attributes
        self.use_images = use_images
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_encoder_dim = image_encoder_dim
        self.fusion_method = fusion_method


        self.config = MambaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.embedding_size,
            state_size=self.state_size,
            num_hidden_layers=self.num_hidden_layers,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            pad_token_id=self.padding_idx,
            bos_token_id=self.cls_idx,
            eos_token_id=self.padding_idx,
            use_mambapy=self.use_mambapy,
        )
        self.embeddings = MambaEmbeddingsForCEHR(
            config=self.config,
            type_vocab_size=self.type_vocab_size,
            max_num_visits=self.max_num_visits,
            time_embeddings_size=self.time_embeddings_size,
            visit_order_size=self.visit_order_size,
            hidden_dropout_prob=self.dropout_prob,
        )
        
        # Image components (only initialized if use_images=True)
        if self.use_images:
            self._setup_image_encoder(
                image_size, 
                patch_size, 
                image_encoder_dim,
                image_encoder_depth,
                image_encoder_heads
            )
            # Project image embeddings to match text embedding dimension
            self.image_projection = nn.Linear(image_encoder_dim, embedding_size)
            
            # Fusion layer (if using MLP fusion)
            if fusion_method == "mlp":
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(embedding_size, embedding_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(embedding_size * 4, embedding_size),
                )
            
            # Layer norm after fusion
            self.fusion_norm = nn.LayerNorm(embedding_size)
                
        
        # Initialize weights and apply final processing
        self.post_init()

        # Mamba has its own initialization
        self.model = MambaForCausalLM(config=self.config)

    def _init_weights(self, module: torch.nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _setup_image_encoder(
        self, 
        image_size, 
        patch_size, 
        dim, 
        depth, 
        heads
    ):
        """Set up the Vision Transformer encoder for images."""
        from zeta.structs import Encoder, ViTransformerWrapper
        
        self.image_encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
            ),
        )
        
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Image embeddings of shape (batch_size, seq_len, embedding_size)
        """
        if not self.use_images:
            raise ValueError("Image encoding is not enabled. Set use_images=True")
        
        # Encode image through ViT
        image_embeds = self.image_encoder(images, return_embeddings=True)
        
        # Project to match text embedding dimension
        image_embeds = self.image_projection(image_embeds)
        
        return image_embeds

    def fuse_embeddings(
        self, 
        ehr_embeds: torch.Tensor, 
        image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and image embeddings.
        
        Args:
            text_embeds: Text embeddings (batch_size, text_seq_len, dim)
            image_embeds: Image embeddings (batch_size, image_seq_len, dim)
            
        Returns:
            Fused embeddings
        """
        if self.fusion_method == "add":
            # Average pool image embeddings to match text length if needed
            if image_embeds.shape[1] != ehr_embeds.shape[1]:
                image_embeds = torch.nn.functional.adaptive_avg_pool1d(
                    image_embeds.transpose(1, 2), 
                    ehr_embeds.shape[1]
                ).transpose(1, 2)
            fused = text_embeds + image_embeds
            
        elif self.fusion_method == "concat":
            # Concatenate along sequence dimension
            fused = torch.cat([text_embeds, image_embeds], dim=1)
            
        elif self.fusion_method == "mlp":
            # Use MLP to fuse
            if image_embeds.shape[1] != ehr_embeds.shape[1]:
                image_embeds = torch.nn.functional.adaptive_avg_pool1d(
                    image_embeds.transpose(1, 2), 
                    ehr_embeds.shape[1]
                ).transpose(1, 2)
            combined = text_embeds + image_embeds
            fused = self.fusion_mlp(combined) + text_embeds  # Residual connection
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Apply layer norm
        fused = self.fusion_norm(fused)
        
        return fused

    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)


    def forward(
        self,
        inputs: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments = inputs
        inputs_embeds = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )

        if labels is None:
            labels = concept_ids

        return self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Train model on training dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"train_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Any:
        """Evaluate model on validation dataset."""
        inputs = (
            batch["concept_ids"],
            batch["type_ids"],
            batch["time_stamps"],
            batch["ages"],
            batch["visit_orders"],
            batch["visit_segments"],
        )
        labels = batch["labels"]

        # Ensure use of mixed precision
        with autocast():
            loss = self(
                inputs,
                labels=labels,
                return_dict=True,
            ).loss

        (current_lr,) = self.lr_schedulers().get_last_lr()
        self.log_dict(
            dictionary={"val_loss": loss, "lr": current_lr},
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=n_decay_steps,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
