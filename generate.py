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
from lm_utils import choose_device, generate_tokens
from tokenizer import Tokenizer


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
        generated_tokens = generate_tokens(
            model,
            starting_tokens,
            tokens_to_generate,
            temperature,
            device,
            context_length=CONTEXT_LENGTH,
            end_of_text=END_OF_TEXT,
        )

    print(tokenizer.decode(generated_tokens, errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=TOKENS_TO_GENERATE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    arguments = parser.parse_args()

    main(arguments.prompt, arguments.tokens, arguments.temperature)
