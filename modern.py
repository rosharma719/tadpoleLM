"""Compact PyTorch version of the model, training loop, and generation."""

import argparse
from pathlib import Path

import torch

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
from gpt import Transformer, TransformerConfig
from lm_utils import (
    choose_device,
    cosine_learning_rate,
    estimate_random_losses,
    generate_tokens,
    next_token_loss,
    random_next_token_batch,
)
from tokenizer import Tokenizer


# This implementation has different state-dict names than model.py, so keep its
# checkpoint separate even though the architecture and tensor sizes are equal.
CHECKPOINT_PATH = "modern_model.pt"

MODEL_CONFIG = TransformerConfig(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_size=EMBEDDING_SIZE,
    number_of_heads=NUMBER_OF_HEADS,
    number_of_blocks=NUMBER_OF_BLOCKS,
    feed_forward_multiplier=FEED_FORWARD_MULTIPLIER,
    dropout=DROPOUT,
    weight_init_std=WEIGHT_INIT_STD,
)


def learning_rate_multiplier(step: int) -> float:
    return cosine_learning_rate(
        step,
        LEARNING_RATE,
        MINIMUM_LEARNING_RATE,
        WARMUP_STEPS,
        LEARNING_RATE_DECAY_STEPS,
    ) / LEARNING_RATE


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
        end_of_text=END_OF_TEXT,
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

    model = Transformer(MODEL_CONFIG).to(device)
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
            training_loss, validation_loss = estimate_random_losses(
                model,
                training_data,
                validation_data,
                BATCH_SIZE,
                CONTEXT_LENGTH,
                EVALUATION_BATCHES,
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

        inputs, targets = random_next_token_batch(
            training_data, BATCH_SIZE, CONTEXT_LENGTH, device
        )
        loss = next_token_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    print(f"best validation:   {best_validation_loss:.4f}")
    print(f"saved:             {CHECKPOINT_PATH}")


def generate(prompt: str, count: int, temperature: float) -> None:
    device = choose_device()
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    model = Transformer(MODEL_CONFIG).to(device)
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
        end_of_text=END_OF_TEXT,
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
