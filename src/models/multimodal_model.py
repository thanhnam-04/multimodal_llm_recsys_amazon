import torch
import torch.nn as nn
from typing import Optional, Tuple

class MultiModalGPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        image_embedding_dim: int = 512  # CLIP ViT-B/32 image embedding dimension
    ):
        """
        Initialize the multi-modal GPT model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            d_model (int): Dimension of the model
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout rate
            image_embedding_dim (int): Dimension of image embeddings
        """
        super().__init__()
        
        # Text embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Image projection layer
        self.image_projection = nn.Linear(image_embedding_dim, d_model)
        
        # Multi-modal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            image_embeddings (Optional[torch.Tensor]): Image embeddings
            attention_mask (Optional[torch.Tensor]): Attention mask
            
        Returns:
            torch.Tensor: Output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Get text embeddings
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_embeddings = self.position_embedding(position_ids).unsqueeze(0)
        text_embeddings = token_embeddings + position_embeddings
        
        # Process image embeddings if available
        if image_embeddings is not None:
            # Project image embeddings to model dimension
            projected_images = self.image_projection(image_embeddings)
            
            # Concatenate text and image embeddings
            combined_embeddings = torch.cat([text_embeddings, projected_images], dim=1)
            
            # Fuse embeddings
            fused_embeddings = self.fusion_layer(combined_embeddings)
        else:
            fused_embeddings = text_embeddings
            
        # Apply dropout
        fused_embeddings = self.dropout(fused_embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create causal mask for transformer
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Apply transformer layers
        hidden_states = fused_embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                src_key_padding_mask=~attention_mask.bool(),
                src_mask=causal_mask
            )
            
        # Get output logits
        logits = self.output_layer(hidden_states)
        
        return logits 