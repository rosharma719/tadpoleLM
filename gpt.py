"""Compact, configurable GPT used by modern.py and cloud_train.py."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int = 4096       # Number of token IDs, including EOT.
    context_length: int = 256    # Maximum tokens visible in one sequence.
    embedding_size: int = 384    # Width C of the residual stream.
    number_of_heads: int = 6     # Attention heads; each gets C/H features.
    number_of_blocks: int = 6    # Repeated attention + feed-forward blocks.
    feed_forward_multiplier: int = 4  # FFN hidden width relative to C.
    dropout: float = 0.1         # Training-only regularization probability.
    weight_init_std: float = 0.02  # Standard deviation of initial weights.


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        width = config.embedding_size
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=config.number_of_heads,
            dropout=config.dropout,
            bias=False,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(config.dropout)

        hidden_size = config.feed_forward_multiplier * width
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + self.attention_output_dropout(attention_output)
        return x + self.feed_forward(self.feed_forward_norm(x))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embedding_size
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_size
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.number_of_blocks)
        )
        self.final_norm = nn.LayerNorm(config.embedding_size)
        self.language_model_head = nn.Linear(
            config.embedding_size, config.vocab_size, bias=False
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

        self.apply(self._initialize_weights)
        # MultiheadAttention owns its combined QKV matrix directly, rather than
        # putting it inside an nn.Linear module visited by self.apply().
        for block in self.blocks:
            nn.init.normal_(
                block.attention.in_proj_weight,
                mean=0.0,
                std=config.weight_init_std,
            )
        self.language_model_head.weight = self.token_embedding.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.weight_init_std,
            )
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"maximum sequence length is {self.config.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in self.blocks:
            x = block(x, mask)
        return self.language_model_head(self.final_norm(x))
