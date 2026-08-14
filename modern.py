"""Compact PyTorch version of the model, training loop, and generation."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from config import (
    AUGMENTED_CORPUS_PATH,
    BATCH_SIZE,
    CONTEXT_LENGTH,
    CORPUS_PATH,
    DEFAULT_PROMPT,
    DROPOUT,
    EMBEDDING_SIZE,
    END_OF_TEXT,
    EVALUATION_BATCHES,
    EVALUATION_INTERVAL,
    FEED_FORWARD_MULTIPLIER,
    LEARNING_RATE,
    LEARNING_RATE_DECAY_STEPS,
    MINIMUM_LEARNING_RATE,
    NUMBER_OF_BLOCKS,
    NUMBER_OF_HEADS,
    RANDOM_SEED,
    TEMPERATURE,
    TOKENIZER_PATH,
    TOKENS_TO_GENERATE,
    TRAINING_FRACTION,
    TRAINING_SAMPLE_TOKENS,
    TRAINING_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    WEIGHT_INIT_STD,
)
from tokenizer import Tokenizer


# This implementation has different state-dict names than model.py, so keep its
# checkpoint separate even though the architecture and tensor sizes are equal.
CHECKPOINT_PATH = "modern_model.pt"


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.attention = nn.MultiheadAttention(
            embed_dim=EMBEDDING_SIZE,
            num_heads=NUMBER_OF_HEADS,
            dropout=DROPOUT,
            bias=False,
            batch_first=True,
        )
        self.attention_output_dropout = nn.Dropout(DROPOUT)

        hidden_size = FEED_FORWARD_MULTIPLIER * EMBEDDING_SIZE
        self.feed_forward_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, EMBEDDING_SIZE),
            nn.Dropout(DROPOUT),
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
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.embedding_dropout = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList(
            TransformerBlock() for _ in range(NUMBER_OF_BLOCKS)
        )
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.language_model_head = nn.Linear(
            EMBEDDING_SIZE,
            VOCAB_SIZE,
            bias=False,
        )

        # True means "do not attend here." persistent=False keeps this derived
        # mask out of the checkpoint; it is rebuilt with the model architecture.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

        self.apply(self.initialize_weights)
        # MultiheadAttention stores its combined QKV matrix directly rather than
        # inside nn.Linear, so initialize that one parameter explicitly.
        for block in self.blocks:
            nn.init.normal_(
                block.attention.in_proj_weight,
                mean=0.0,
                std=WEIGHT_INIT_STD,
            )

        self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=WEIGHT_INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_length = tokens.shape[1]
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(f"maximum sequence length is {CONTEXT_LENGTH}")

        positions = torch.arange(sequence_length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        for block in self.blocks:
            x = block(x, causal_mask)

        return self.language_model_head(self.final_norm(x))


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(data: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - CONTEXT_LENGTH, size=(BATCH_SIZE,))
    inputs = torch.stack(
        [data[start : start + CONTEXT_LENGTH] for start in starts]
    )
    targets = torch.stack(
        [data[start + 1 : start + CONTEXT_LENGTH + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def learning_rate_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    if step >= LEARNING_RATE_DECAY_STEPS:
        return MINIMUM_LEARNING_RATE / LEARNING_RATE

    progress = (step - WARMUP_STEPS) / (
        LEARNING_RATE_DECAY_STEPS - WARMUP_STEPS
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = MINIMUM_LEARNING_RATE / LEARNING_RATE
    return minimum + cosine * (1.0 - minimum)


@torch.inference_mode()
def estimate_loss(
    model: Transformer,
    training_data: torch.Tensor,
    validation_data: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    losses = {}

    for name, data in (("train", training_data), ("validation", validation_data)):
        measurements = []
        for _ in range(EVALUATION_BATCHES):
            inputs, targets = get_batch(data, device)
            measurements.append(calculate_loss(model(inputs), targets).item())
        losses[name] = sum(measurements) / len(measurements)

    model.train(was_training)
    return losses["train"], losses["validation"]


@torch.inference_mode()
def generate_tokens(
    model: Transformer,
    starting_tokens: list[int],
    count: int,
    temperature: float,
    device: str,
) -> list[int]:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    tokens = torch.tensor([starting_tokens], dtype=torch.long, device=device)
    for _ in range(count):
        logits = model(tokens[:, -CONTEXT_LENGTH:])[:, -1, :]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item() == END_OF_TEXT:
            break

    return tokens[0].tolist()


def print_sample(
    model: Transformer,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    was_training = model.training
    model.eval()
    prompt_tokens = tokenizer.encode(DEFAULT_PROMPT) or [END_OF_TEXT]
    generated = generate_tokens(
        model,
        prompt_tokens,
        TRAINING_SAMPLE_TOKENS,
        TEMPERATURE,
        device,
    )
    print(f"sample ({DEFAULT_PROMPT!r}):")
    print(tokenizer.decode(generated, errors="replace"))
    model.train(was_training)


def train(training_steps: int) -> None:
    torch.manual_seed(RANDOM_SEED)
    device = choose_device()
    tokenizer = Tokenizer.load(TOKENIZER_PATH)

    augmented_path = Path(AUGMENTED_CORPUS_PATH)
    corpus_path = augmented_path if augmented_path.exists() else Path(CORPUS_PATH)
    corpus = corpus_path.read_text(encoding="utf-8")
    all_tokens = torch.tensor(
        tokenizer.encode(corpus, add_end=True),
        dtype=torch.long,
    )

    split = int(TRAINING_FRACTION * len(all_tokens))
    training_data = all_tokens[:split]
    validation_data = all_tokens[split:]

    model = Transformer().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_multiplier,
    )

    print(f"device:            {device}")
    print(f"corpus:            {corpus_path}")
    print(f"training tokens:   {len(training_data):,}")
    print(f"validation tokens: {len(validation_data):,}")
    print(f"parameters:        {sum(p.numel() for p in model.parameters()):,}")
    print(f"checkpoint:        {CHECKPOINT_PATH}")

    best_validation_loss = float("inf")
    for step in range(training_steps):
        if step % EVALUATION_INTERVAL == 0 or step == training_steps - 1:
            training_loss, validation_loss = estimate_loss(
                model,
                training_data,
                validation_data,
                device,
            )
            saved_best = validation_loss < best_validation_loss
            if saved_best:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:4d} | train loss {training_loss:.4f} | "
                f"validation loss {validation_loss:.4f} | lr {current_lr:.2e}"
                f"{' | saved best' if saved_best else ''}"
            )
            print_sample(model, tokenizer, device)

        inputs, targets = get_batch(training_data, device)
        loss = calculate_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"best validation:   {best_validation_loss:.4f}")
    print(f"saved:             {CHECKPOINT_PATH}")


def generate(prompt: str, count: int, temperature: float) -> None:
    device = choose_device()
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    model = Transformer().to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    starting_tokens = tokenizer.encode(prompt) or [END_OF_TEXT]
    generated = generate_tokens(
        model,
        starting_tokens,
        count,
        temperature,
        device,
    )
    print(tokenizer.decode(generated, errors="replace"))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=TRAINING_STEPS)

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    generate_parser.add_argument("--tokens", type=int, default=TOKENS_TO_GENERATE)
    generate_parser.add_argument("--temperature", type=float, default=TEMPERATURE)

    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments.steps)
    else:
        generate(arguments.prompt, arguments.tokens, arguments.temperature)


if __name__ == "__main__":
    main()
