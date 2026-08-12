"""The first half of a tiny transformer: embeddings and self-attention."""

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
        return weights @ values


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.attention = SelfAttention()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert token IDs into context-aware vectors."""

        sequence_length = tokens.shape[1]
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(f"maximum sequence length is {CONTEXT_LENGTH}")

        positions = torch.arange(sequence_length, device=tokens.device)

        token_vectors = self.token_embedding(tokens)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors

        return self.attention(x)


if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load("tokenizer.json")
    token_ids = tokenizer.encode("i think")

    # Shape is (batch, time): one example containing two tokens.
    tokens = torch.tensor([token_ids])

    model = TinyTransformer()
    output = model(tokens)

    print("token IDs:    ", token_ids)
    print("input shape:  ", tuple(tokens.shape))
    print("output shape: ", tuple(output.shape))
