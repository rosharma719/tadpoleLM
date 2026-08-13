"""A tiny, one-block transformer language model."""

import math

import torch
from torch import nn


VOCAB_SIZE = 1024
CONTEXT_LENGTH = 128
EMBEDDING_SIZE = 128


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Each token vector is transformed into a query, key, and value.
        self.query = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.key = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)

        # Transform the gathered information before writing it back to x.
        self.output = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Let every token gather information from itself and earlier tokens."""

        sequence_length = x.shape[1]

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Compare every query with every key. Dividing keeps the scores from
        # becoming too large as the vectors get wider.
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(EMBEDDING_SIZE)

        # Row i may look only at columns 0 through i. Future scores become
        # negative infinity, so softmax turns them into zero.
        allowed = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device)
        ).bool()
        scores = scores.masked_fill(~allowed, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        context = weights @ values

        return self.output(context)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        # Give each token more room to compute, then return it to the model size.
        self.expand = nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE)
        self.activation = nn.GELU()
        self.contract = nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.activation(x)
        return self.contract(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.attention = SelfAttention()

        self.feed_forward_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.feed_forward = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each sublayer proposes an update; the residual connection preserves x.
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.block = TransformerBlock()
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)

        # Convert each final token vector into one score per vocabulary token.
        self.language_model_head = nn.Linear(
            EMBEDDING_SIZE,
            VOCAB_SIZE,
            bias=False,
        )

        # Start all learned matrices with small values so initial logits and
        # gradients stay in a useful range.
        self.apply(self.initialize_weights)

        # Use the same matrix to read token embeddings and score output tokens.
        self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token scores for every position in every example."""

        sequence_length = tokens.shape[1]
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(f"maximum sequence length is {CONTEXT_LENGTH}")

        positions = torch.arange(sequence_length, device=tokens.device)

        token_vectors = self.token_embedding(tokens)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors

        x = self.block(x)
        x = self.final_norm(x)
        logits = self.language_model_head(x)

        return logits


if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load("tokenizer.json")
    token_ids = tokenizer.encode("i think")

    # Shape is (batch, time): one example containing two tokens.
    tokens = torch.tensor([token_ids])

    model = TinyTransformer()
    logits = model(tokens)

    print("token IDs:    ", token_ids)
    print("input shape:  ", tuple(tokens.shape))
    print("logits shape: ", tuple(logits.shape))
