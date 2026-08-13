"""Train the tiny transformer to predict the next token."""

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from config import (
    AUGMENTED_CORPUS_PATH,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    CORPUS_PATH,
    DEFAULT_PROMPT,
    END_OF_TEXT,
    EVALUATION_BATCHES,
    EVALUATION_INTERVAL,
    LEARNING_RATE,
    LEARNING_RATE_DECAY_STEPS,
    MINIMUM_LEARNING_RATE,
    RANDOM_SEED,
    TOKENIZER_PATH,
    TEMPERATURE,
    TRAINING_FRACTION,
    TRAINING_SAMPLE_TOKENS,
    TRAINING_STEPS,
    VOCAB_SIZE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from generate import generate
from model import TinyTransformer
from tokenizer import Tokenizer


def choose_device() -> str:
    # MPS is PyTorch's backend for Apple Silicon GPUs; tensors and the model
    # must live on the same device before they can participate in operations.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_batch(data: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Select random input windows and their one-token-shifted targets."""

    # Each random start creates one independent sequence. Stacking B windows
    # produces (B,T); batching parallelizes work but does not connect sequences.
    # torch.randint(high, size=(B,)) creates B random integer starting indices.
    starts = torch.randint(len(data) - CONTEXT_LENGTH, size=(BATCH_SIZE,))

    # If one window is [10,20,30,40], its input is [10,20,30] and target is
    # [20,30,40]. The model therefore learns one next-token target per position.
    # Each slice has shape (T). torch.stack creates a new leading batch axis,
    # turning B separate slices into one tensor of shape (B,T).
    inputs = torch.stack(
        [data[start : start + CONTEXT_LENGTH] for start in starts]
    )
    targets = torch.stack(
        [data[start + 1 : start + CONTEXT_LENGTH + 1] for start in starts]
    )

    # Tensor.to(device) copies CPU batch data to MPS/CUDA when necessary.
    return inputs.to(device), targets.to(device)


def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compare every vocabulary-score vector with its correct next token."""

    batch_size, sequence_length, vocabulary_size = logits.shape

    # Cross-entropy expects one row of V scores per target. Flatten B and T so
    # (B,T,V) becomes (B*T,V), giving B*T simultaneous predictions.
    # Tensor.reshape returns the same values viewed as a different shape.
    logits = logits.reshape(batch_size * sequence_length, vocabulary_size)
    targets = targets.reshape(batch_size * sequence_length)

    # Pass raw logits, not probabilities. cross_entropy performs a numerically
    # stable log_softmax internally, then penalizes the correct target token.
    # Applying softmax first would both duplicate that work and be less stable.
    # F is torch.nn.functional: stateless tensor operations rather than layer
    # objects stored on the model. There are no learned loss parameters here.
    return F.cross_entropy(logits, targets)


def learning_rate_at_step(step: int) -> float:
    """Warm up linearly, then reduce the learning rate along a cosine curve."""

    # Early random gradients can be destructive, so the first WARMUP_STEPS
    # updates grow gradually from a tiny value to LEARNING_RATE.
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    # Once decay is complete, hold the small minimum learning rate steady.
    if step >= LEARNING_RATE_DECAY_STEPS:
        return MINIMUM_LEARNING_RATE

    # progress moves from 0 to 1 through the decay period. cos(pi*progress)
    # moves smoothly from 1 to -1, so multiplier moves from 1 to 0.
    progress = (step - WARMUP_STEPS) / (
        LEARNING_RATE_DECAY_STEPS - WARMUP_STEPS
    )
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MINIMUM_LEARNING_RATE + multiplier * (
        LEARNING_RATE - MINIMUM_LEARNING_RATE
    )


# Disable construction of the backward graph while merely measuring loss.
@torch.no_grad()
def estimate_loss(
    model: TinyTransformer,
    training_data: torch.Tensor,
    validation_data: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """Measure average loss without calculating gradients or changing weights."""

    # eval() disables dropout; it does not turn gradients off by itself.
    model.eval()
    losses = {}

    for name, data in [("train", training_data), ("validation", validation_data)]:
        measurements = []

        for _ in range(EVALUATION_BATCHES):
            inputs, targets = get_batch(data, device)
            logits = model(inputs)
            # .item() extracts a one-element tensor as an ordinary Python float.
            measurements.append(calculate_loss(logits, targets).item())

        losses[name] = sum(measurements) / len(measurements)

    # Restore training behavior so dropout is active during parameter updates.
    model.train()
    return losses["train"], losses["validation"]


@torch.no_grad()
def print_sample(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    device: str,
) -> None:
    """Print one short response showing what the current model has learned."""

    # Generation is inference, so disable training-only behavior while sampling.
    model.eval()
    prompt_tokens = tokenizer.encode(DEFAULT_PROMPT)
    if not prompt_tokens:
        prompt_tokens = [END_OF_TEXT]

    generated_tokens = generate(
        model,
        prompt_tokens,
        TRAINING_SAMPLE_TOKENS,
        TEMPERATURE,
        device,
    )

    print(f"sample ({DEFAULT_PROMPT!r}):")
    print(tokenizer.decode(generated_tokens, errors="replace"))
    model.train()


def main(training_steps: int) -> None:
    torch.manual_seed(RANDOM_SEED)
    device = choose_device()

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    # augment.py creates the larger corpus. Until then, training still works on
    # the original notes. The printed path below makes this choice explicit.
    augmented_path = Path(AUGMENTED_CORPUS_PATH)
    corpus_path = augmented_path if augmented_path.exists() else Path(CORPUS_PATH)
    corpus = corpus_path.read_text(encoding="utf-8")
    # dtype=torch.long means signed 64-bit integers, the index type expected by
    # nn.Embedding and class targets expected by cross_entropy.
    all_tokens = torch.tensor(
        tokenizer.encode(corpus, add_end=True),
        dtype=torch.long,
    )

    split = int(TRAINING_FRACTION * len(all_tokens))
    training_data = all_tokens[:split]
    validation_data = all_tokens[split:]

    # TinyTransformer() constructs parameters on CPU; Module.to(device) moves
    # every registered parameter and buffer to the Apple GPU/other backend.
    model = TinyTransformer().to(device)
    # model.parameters() recursively yields all trainable nn.Parameter tensors.
    # The optimizer keeps update state for them but is not part of model.forward.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device:           {device}")
    print(f"corpus:           {corpus_path}")
    print(f"training tokens:  {len(training_data):,}")
    print(f"validation tokens: {len(validation_data):,}")
    print(f"parameters:       {parameter_count:,}")

    # Start above every possible real loss. Whenever validation improves, the
    # checkpoint on disk is replaced. A later overfit model cannot overwrite it.
    best_validation_loss = float("inf")

    for step in range(training_steps):
        current_learning_rate = learning_rate_at_step(step)
        # param_groups is a list because an optimizer may use different settings
        # for different parameters. We have one group, but update it generically.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

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
                # state_dict() maps parameter names to tensors. We save only the
                # model weights needed by generate.py, not the whole Python object.
                torch.save(model.state_dict(), CHECKPOINT_PATH)

            print(
                f"step {step:4d} | "
                f"train loss {training_loss:.4f} | "
                f"validation loss {validation_loss:.4f}"
                f" | lr {current_learning_rate:.2e}"
                f"{' | saved best' if saved_best else ''}"
            )
            print_sample(model, tokenizer, device)

        inputs, targets = get_batch(training_data, device)
        logits = model(inputs)
        loss = calculate_loss(logits, targets)

        # PyTorch accumulates gradients by default, so clear old ones first.
        optimizer.zero_grad(set_to_none=True)

        # loss is a scalar tensor with a recorded computation graph. backward()
        # walks that graph in reverse and fills each Parameter's .grad tensor.
        loss.backward()

        # AdamW reads .grad and mutates the parameter tensors in place.
        optimizer.step()

    print(f"best validation:  {best_validation_loss:.4f}")
    print(f"saved:            {CHECKPOINT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TRAINING_STEPS)
    arguments = parser.parse_args()

    main(arguments.steps)
