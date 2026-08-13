"""Generate text one token at a time from a trained checkpoint."""

import argparse

import torch

from config import (
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    DEFAULT_PROMPT,
    END_OF_TEXT,
    TEMPERATURE,
    TOKENIZER_PATH,
    TOKENS_TO_GENERATE,
)
from model import TinyTransformer
from tokenizer import Tokenizer


def choose_device() -> str:
    # These Boolean queries ask which PyTorch acceleration backend is usable;
    # "mps" selects the Apple Silicon GPU on this machine.
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(
    model: TinyTransformer,
    starting_tokens: list[int],
    tokens_to_generate: int,
    temperature: float,
    device: str,
) -> list[int]:
    """Repeatedly sample one token from the model's final-position logits."""

    # torch.tensor creates a (1,T) integer tensor directly on the chosen device.
    tokens = torch.tensor([starting_tokens], dtype=torch.long, device=device)
    # The outer list creates batch size B=1. Inference can also batch prompts,
    # but autoregressive generation commonly grows one sequence at a time.

    for _ in range(tokens_to_generate):
        # Position embeddings only cover the most recent CONTEXT_LENGTH tokens.
        # Tensor slicing uses [batch, time]. ':' keeps the only batch; the
        # negative slice keeps at most the final CONTEXT_LENGTH token IDs.
        current_context = tokens[:, -CONTEXT_LENGTH:]
        logits = model(current_context)

        # logits is (1,T,V). [:,-1,:] means every batch item, final position,
        # every vocabulary score, yielding (1,V) for the next unknown token.
        next_token_logits = logits[:, -1, :] / temperature

        # Unlike training loss, sampling needs actual probabilities, so this is
        # where we explicitly apply softmax before drawing one token ID.
        probabilities = torch.softmax(next_token_logits, dim=-1)
        # torch.multinomial samples one column/token ID from each probability row,
        # returning shape (B,1), here concretely (1,1).
        next_token = torch.multinomial(probabilities, num_samples=1)

        # torch.cat joins tensors along an existing axis. dim=1 appends the new
        # token to time, changing (1,T) into (1,T+1).
        tokens = torch.cat((tokens, next_token), dim=1)

        if next_token.item() == END_OF_TEXT:
            break

    return tokens[0].tolist()


def main(prompt: str, tokens_to_generate: int, temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    device = choose_device()
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    starting_tokens = tokenizer.encode(prompt)

    if not starting_tokens:
        starting_tokens = [END_OF_TEXT]

    model = TinyTransformer().to(device)
    # torch.load reads the name->tensor mapping. map_location places tensors on
    # this device; load_state_dict copies them into the constructed model.
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()  # Disable dropout so inference uses every learned activation.

    # Generation needs no gradients because it does not update model parameters.
    with torch.no_grad():
        generated_tokens = generate(
            model,
            starting_tokens,
            tokens_to_generate,
            temperature,
            device,
        )

    print(tokenizer.decode(generated_tokens, errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=TOKENS_TO_GENERATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    arguments = parser.parse_args()

    main(arguments.prompt, arguments.tokens, arguments.temperature)
