from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import autocast

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


class ICUMultimodalMambda(nn.Module):
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
        use_images=True,
        dropout_prob=0.1,
        padding_idx=257,
        cls_idx=5,
        use_mambapy=False,
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
        self.dropout_prob = dropout_prob
        self.padding_idx=padding_idx
        self.cls_idx=cls_idx

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
            use_mambapy=use_mambapy,
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
    
    def post_init(self) -> None:
        """Apply weight initialization."""
        self.apply(self._init_weights)

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
            ehr_embeds: Text embeddings (batch_size, text_seq_len, dim)
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
            fused = ehr_embeds + image_embeds
            
        elif self.fusion_method == "concat":
            # Concatenate along sequence dimension
            fused = torch.cat([ehr_embeds, image_embeds], dim=1)
            
        elif self.fusion_method == "mlp":
            # Use MLP to fuse
            if image_embeds.shape[1] != ehr_embeds.shape[1]:
                image_embeds = torch.nn.functional.adaptive_avg_pool1d(
                    image_embeds.transpose(1, 2), 
                    ehr_embeds.shape[1]
                ).transpose(1, 2)
            combined = ehr_embeds + image_embeds
            fused = self.fusion_mlp(combined) + ehr_embeds  # Residual connection
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Apply layer norm
        fused = self.fusion_norm(fused)
        
        return fused

    def forward(
        self,
        inputs,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], MambaCausalLMOutput]:
        """Forward pass for the model."""
        concept_ids, type_ids, time_stamps, ages, visit_orders, visit_segments, images = inputs
        inputs = self.embeddings(
            input_ids=concept_ids,
            token_type_ids_batch=type_ids,
            time_stamps=time_stamps,
            ages=ages,
            visit_orders=visit_orders,
            visit_segments=visit_segments,
        )
        

        if self.use_images:
            inputs=self.fuse_embeddings(inputs,images)
        

        if labels is None:
            labels = concept_ids

        
        return self.model(
            inputs_embeds=inputs,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )