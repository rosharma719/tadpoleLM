"""Train the tiny transformer to predict the next token."""

import argparse
from pathlib import Path

import torch

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
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from lm_utils import (
    choose_device,
    cosine_learning_rate,
    estimate_random_losses,
    generate_tokens,
    next_token_loss,
    random_next_token_batch,
)
from model import TinyTransformer
from tokenizer import Tokenizer


def learning_rate_at_step(step: int) -> float:
    """Warm up linearly, then reduce the learning rate along a cosine curve."""
    return cosine_learning_rate(
        step,
        LEARNING_RATE,
        MINIMUM_LEARNING_RATE,
        WARMUP_STEPS,
        LEARNING_RATE_DECAY_STEPS,
    )


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

    generated_tokens = generate_tokens(
        model,
        prompt_tokens,
        TRAINING_SAMPLE_TOKENS,
        TEMPERATURE,
        device,
        context_length=CONTEXT_LENGTH,
        end_of_text=END_OF_TEXT,
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

        inputs, targets = random_next_token_batch(
            training_data, BATCH_SIZE, CONTEXT_LENGTH, device
        )
        logits = model(inputs)
        loss = next_token_loss(logits, targets)

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
