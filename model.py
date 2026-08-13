"""A tiny transformer language model.

PyTorch convention used throughout this file
------------------------------------------------
Every class below inherits from ``nn.Module``. An ``nn.Module`` object is
callable: writing ``layer(x)`` runs PyTorch's ``layer.__call__(x)``, which in
turn runs the layer's ``forward(x)`` method while preserving hooks and autograd
bookkeeping. For example, after ``self.query = nn.Linear(...)``:

    self.query(x)             # preferred PyTorch syntax
    self.query.forward(x)     # underlying computation, but do not call directly

Creating ``nn.Linear(...)`` happens once in ``__init__`` and creates learned
parameters. Calling ``self.query(x)`` happens on every forward pass and reuses
those same parameters.

Shape symbols and representation spaces
---------------------------------------
B = batch size; independent sequences processed in parallel.
T = sequence length; token positions in each sequence.
C = embedding/residual width; currently 256 features per token.
H = attention heads; currently 4 independent attention patterns.
D = per-head width, C/H; currently 64 features per head.
F = feed-forward hidden width, 4C; currently 1,024 features per token.
V = vocabulary size; currently 1,024 possible token IDs.

The model moves through several different spaces:

    token-ID space       (B,T)       discrete integers in [0, V)
    residual space       (B,T,C)     continuous token representations
    per-head space       (B,H,T,D)   Q/K/V features split among heads
    attention space      (B,H,T,T)   token-to-token scores or weights
    FFN hidden space     (B,T,F)     expanded per-token features
    vocabulary space     (B,T,V)     next-token logits
"""

import math

import torch
from torch import nn

from config import (
    CONTEXT_LENGTH,
    DROPOUT,
    EMBEDDING_SIZE,
    FEED_FORWARD_MULTIPLIER,
    NUMBER_OF_BLOCKS,
    NUMBER_OF_HEADS,
    TOKENIZER_PATH,
    VOCAB_SIZE,
    WEIGHT_INIT_STD,
)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        # nn.Module tracks every layer assigned to self, including its trainable
        # tensors, so optimizer/model saving can find them automatically.
        super().__init__()

        if EMBEDDING_SIZE % NUMBER_OF_HEADS != 0:
            raise ValueError("embedding size must divide evenly across attention heads")

        self.head_size = EMBEDDING_SIZE // NUMBER_OF_HEADS

        # Each nn.Linear owns a learned 256x256 matrix. Calling self.query(x)
        # applies that same matrix to every token in every batch example.
        # nn.Linear(C,C) stores weight (C,C). Calling layer(x) applies the same
        # matrix to x's last dimension, preserving (B,T,C) while changing the
        # coordinates from residual space into Q, K, or V feature space.
        self.query = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.key = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)

        # WO, the attention output matrix: mix the rejoined heads before their
        # result is written into the residual stream. This is not the FFN's
        # contraction layer or the final language-model output matrix.
        self.output = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE, bias=False)
        self.attention_dropout = nn.Dropout(DROPOUT)
        self.output_dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Let every token gather information from itself and earlier tokens."""

        # x lives in residual space and has shape (B,T,C), concretely
        # (B,T,256). Tuple unpacking assigns those three dimension sizes.
        batch_size, sequence_length, _ = x.shape

        # First Q, K, V are each (B, T, C). Reshape C into H heads of D=C/H
        # features: (B, T, C) -> (B, T, H, D), currently D=256/4=64.
        # self.query(x) means "call this nn.Linear module with x." It invokes
        # nn.Linear.forward and computes x @ weight.T on every (B,T) vector.
        # Result before reshape: (B,T,C). After reshape: (B,T,H,D), currently
        # (B,T,6,64). reshape changes organization, not numeric values.
        queries = self.query(x).reshape(
            batch_size, sequence_length, NUMBER_OF_HEADS, self.head_size
        )
        keys = self.key(x).reshape(
            batch_size, sequence_length, NUMBER_OF_HEADS, self.head_size
        )
        values = self.value(x).reshape(
            batch_size, sequence_length, NUMBER_OF_HEADS, self.head_size
        )

        # transpose(1, 2) swaps dimensions 1 and 2 without changing their data:
        # (B, T, H, D) -> (B, H, T, D). The batch examples and heads remain
        # independent; subsequent @ multiplies one T-by-D matrix per (B, H).
        # Tensor.transpose(a,b) returns a view with dimensions a and b swapped.
        # It is unrelated to calling a module: this is a tensor rearrangement.
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # keys.transpose(-2, -1) swaps its last two dimensions, T and D:
        # Q (B,H,T,D) @ K^T (B,H,D,T) -> scores (B,H,T,T).
        # D disappears because each query/key dot product sums its D products;
        # the two T dimensions remain because every query token is compared
        # with every key token. Each batch item gets H separate T-by-T tables.
        # ``@`` is Python's matrix-multiplication operator. PyTorch treats B,H
        # as batch dimensions and multiplies the last two dimensions:
        # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T).
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        # Row i may look only at columns 0 through i. Future scores become
        # negative infinity, so softmax turns them into zero.
        # torch.ones(T,T) constructs a new tensor. torch.tril keeps its lower
        # triangle. .bool() is a tensor method converting 1/0 to True/False.
        # allowed lives in mask space (T,T) and broadcasts across B and H.
        allowed = torch.tril(
            torch.ones(sequence_length, sequence_length, device=x.device)
        ).bool()
        # ``~`` means Boolean NOT. masked_fill returns scores with forbidden
        # cells replaced by -inf; it does not modify scores in place because
        # its name lacks PyTorch's conventional trailing underscore.
        scores = scores.masked_fill(~allowed, float("-inf"))

        # These are input-dependent attention weights, not permanent learned
        # parameters. dim=-1 normalizes each query row across its T key tokens.
        # torch.softmax is a tensor function. dim=-1 means the last axis: each
        # query normalizes its T key scores into weights that sum to one.
        # Shape stays (B,H,T,T), now in attention-probability space.
        weights = torch.softmax(scores, dim=-1)

        # nn.Dropout is callable too. model.train() makes it randomly mask
        # weights; model.eval() makes it return them unchanged.
        weights = self.attention_dropout(weights)

        # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D): each query position
        # receives a weighted sum of the value vector at every allowed position.
        context = weights @ values

        # Put H heads back beside one another: (B,H,T,D) -> (B,T,H,D), then
        # flatten H*D back into C, producing (B,T,C), currently C=256.
        context = context.transpose(1, 2)

        # reshape joins H and D because H*D=C: (B,T,H,D) -> (B,T,C).
        # We have now returned from per-head space to residual-width space.
        context = context.reshape(batch_size, sequence_length, EMBEDDING_SIZE)

        # Calling WO mixes the concatenated heads: (B,T,C) -> (B,T,C).
        context = self.output(context)
        return self.output_dropout(context)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        # The FFN handles each token independently: C -> 4C -> C. "contract"
        # is only a readable name for the 4C -> C linear layer; it does not
        # contain or replace attention's separate WO output matrix above.
        hidden_size = FEED_FORWARD_MULTIPLIER * EMBEDDING_SIZE
        self.expand = nn.Linear(EMBEDDING_SIZE, hidden_size)
        self.activation = nn.GELU()
        self.contract = nn.Linear(hidden_size, EMBEDDING_SIZE)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x begins in residual space: (B,T,C), currently (B,T,256).
        # Calling expand applies nn.Linear(C,F): (B,T,C) -> (B,T,F).
        x = self.expand(x)

        # Calling GELU applies one nonlinear scalar function to every element;
        # it owns no learned weights and keeps shape (B,T,F).
        x = self.activation(x)

        # Calling contract applies nn.Linear(F,C), returning to residual width:
        # (B,T,F) -> (B,T,C). It is not attention's WO matrix.
        x = self.contract(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # LayerNorm(C) normalizes the C features of each (batch, token) vector
        # independently and retains shape (B,T,C).
        self.attention_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.attention = MultiHeadSelfAttention()

        self.feed_forward_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.feed_forward = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calls evaluate inside-out: normalize x, calculate an attention update,
        # then add the original x. Both sides have shape (B,T,C).
        # Nested calls run inside-out:
        #   normalized = self.attention_norm(x)  # call LayerNorm.forward
        #   update = self.attention(normalized)   # call attention.forward
        #   x = x + update                        # elementwise residual addition
        x = x + self.attention(self.attention_norm(x))

        # Attention communicates between tokens; the FFN then computes on each
        # token independently. This is another residual update of shape (B,T,C).
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # nn.Embedding(V,C) owns a learned table of shape (V,C). Calling it with
        # integer IDs performs row lookup rather than matrix multiplication.
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)

        # This separate table maps position IDs 0..T-1 into the same C-wide
        # residual space so token and position vectors can be added.
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.embedding_dropout = nn.Dropout(DROPOUT)
        # A plain Python list would not register child-module parameters.
        # nn.ModuleList behaves like a list while exposing all block parameters
        # to model.parameters(), .to(device), state_dict(), train(), and eval().
        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(NUMBER_OF_BLOCKS)]
        )
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)

        # The language-model output matrix is distinct from attention WO and
        # FFN contract. It maps C -> V to produce one raw score per token ID.
        self.language_model_head = nn.Linear(
            EMBEDDING_SIZE,
            VOCAB_SIZE,
            bias=False,
        )

        # Start all learned matrices with small values so initial logits and
        # gradients stay in a useful range.
        # nn.Module.apply recursively calls initialize_weights(child) on this
        # model and every registered submodule. This is unrelated to forward().
        self.apply(self.initialize_weights)

        # This assignment shares one Parameter rather than copying its values:
        # input uses rows as embeddings; output dots hidden states with all rows.
        self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        # @staticmethod means Python does not insert self; apply supplies the
        # module being initialized as the sole argument.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # A trailing underscore means mutate the tensor in place.
            nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token scores for every position in every example."""

        # tokens is (B,T): B independent sequences, each T token IDs long.
        # Layers share parameters across B; attention never mixes batch items.
        sequence_length = tokens.shape[1]
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(f"maximum sequence length is {CONTEXT_LENGTH}")

        # torch.arange(T) creates [0,1,...,T-1]. device=tokens.device prevents
        # illegal CPU/GPU mixing when tokens live on MPS or CUDA.
        positions = torch.arange(sequence_length, device=tokens.device)

        # Embedding lookup turns (B,T) IDs into (B,T,C) floating-point vectors.
        # Calling embeddings performs lookup:
        # token IDs (B,T) -> token vectors (B,T,C)
        # position IDs (T) -> position vectors (T,C)
        position_vectors = self.position_embedding(positions)
        token_vectors = self.token_embedding(tokens)

        # position_vectors is (T,C); broadcasting reuses it for all B examples.
        x = self.embedding_dropout(token_vectors + position_vectors)

        # Each block is itself callable. model(x) -> Module.__call__ -> forward.
        # Every block keeps x in residual space with shape (B,T,C).
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        # (B,T,C) -> (B,T,V). logits[b,t,:] contains V raw scores for the
        # token following position t in independent sequence b.
        logits = self.language_model_head(x)

        return logits


if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    token_ids = tokenizer.encode("i think")

    # Shape is (batch, time): one example containing two tokens.
    # torch.tensor converts Python lists into tensor storage. The outer list
    # makes one batch: [T IDs] -> shape (1,T). Embeddings require integer IDs.
    tokens = torch.tensor([token_ids], dtype=torch.long)

    model = TinyTransformer()
    # TinyTransformer is callable because it inherits nn.Module. This runs
    # TinyTransformer.forward(tokens) plus PyTorch's surrounding bookkeeping.
    logits = model(tokens)

    print("token IDs:    ", token_ids)
    print("input shape:  ", tuple(tokens.shape))
    print("logits shape: ", tuple(logits.shape))
