"""Small operations shared by training and generation entry points.

The model implementations are intentionally separate: model.py spells out the
math for learning, while gpt.py uses compact modern PyTorch. The mechanics here
should behave identically regardless of which model produced the logits.
"""

import math

import torch
from torch.nn import functional as F


def choose_device(cuda_first: bool = False) -> str:
    """Return the best available PyTorch backend for this workload."""
    if cuda_first and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over every (batch, time) next-token prediction."""
    # (B,T,V) -> (B*T,V), while (B,T) -> (B*T). Cross entropy receives raw
    # logits because it performs its own numerically stable log-softmax.
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def random_next_token_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select B random T-token windows and their one-token-shifted targets."""
    starts = torch.randint(len(data) - context_length, size=(batch_size,))
    inputs = torch.stack(
        [data[start : start + context_length] for start in starts]
    )
    targets = torch.stack(
        [data[start + 1 : start + context_length + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.inference_mode()
def estimate_random_losses(
    model: torch.nn.Module,
    training_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    context_length: int,
    batches: int,
    device: str | torch.device,
) -> tuple[float, float]:
    """Average dropout-free loss over random train and validation batches."""
    was_training = model.training
    model.eval()
    results = []
    for data in (training_data, validation_data):
        measurements = []
        for _ in range(batches):
            inputs, targets = random_next_token_batch(
                data, batch_size, context_length, device
            )
            measurements.append(next_token_loss(model(inputs), targets).item())
        results.append(sum(measurements) / len(measurements))
    model.train(was_training)
    return results[0], results[1]


def cosine_learning_rate(
    step: int,
    peak: float,
    minimum: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Warm up linearly, decay with a cosine, then remain at ``minimum``."""
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    if step >= decay_steps:
        return minimum

    progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + multiplier * (peak - minimum)


@torch.inference_mode()
def generate_tokens(
    model: torch.nn.Module,
    starting_tokens: list[int],
    count: int,
    temperature: float,
    device: str | torch.device,
    context_length: int | None = None,
    end_of_text: int | None = None,
) -> list[int]:
    """Autoregressively sample token IDs from any model returning (B,T,V)."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if context_length is None:
        context_length = model.config.context_length

    tokens = torch.tensor([starting_tokens], dtype=torch.long, device=device)
    for _ in range(count):
        logits = model(tokens[:, -context_length:])[:, -1, :]
        probabilities = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if end_of_text is not None and next_token.item() == end_of_text:
            break

    return tokens[0].tolist()
