"""Read the residual stream after every layer, not just the last one.

The logit lens
--------------
``TinyTransformer.forward`` only converts the residual stream into vocabulary
scores once, at the very end: ``language_model_head(final_norm(x))``. But x has
shape (B,T,C) after *every* block, so that same unembedding can be applied to
the half-finished residual stream too. Doing so asks each layer a question:

    "if the model had to guess the next token right now, what would it say?"

Because the language-model head is tied to the token embedding table, this is
literally asking which token embeddings the current residual vector points
toward. Early layers usually answer with generic high-frequency tokens; the
prediction sharpens as attention and the feed-forward networks write into the
stream. That progression is the "lens."

This is an approximation, not ground truth. Blocks 0-2 were never trained to be
readable through final_norm, so their decoded guesses are suggestive rather than
authoritative. The final row is exact: it is the model's real output.
"""

import argparse

import torch

from config import (
    CHECKPOINT_PATH,
    CONTEXT_LENGTH,
    DEFAULT_PROMPT,
    END_OF_TEXT,
    RANDOM_SEED,
    TEMPERATURE,
    TOKENIZER_PATH,
)
from generate import choose_device
from model import TinyTransformer
from tokenizer import Tokenizer


TOKEN_COLUMN_WIDTH = 11  # Characters reserved for one token's printed form.
LAYER_COLUMN_WIDTH = 14  # Characters reserved for the row label.


def capture_residual_stream(
    model: TinyTransformer,
    tokens: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    """Run the model once, keeping the residual stream after every stage.

    Forward hooks are used instead of re-implementing forward() so this stays
    correct even if the architecture changes. register_forward_hook asks
    PyTorch to call our function with (module, inputs, output) each time that
    module runs; we ignore the first two and keep the output, which is the
    (B,T,C) residual stream leaving that stage.
    """

    captured: list[tuple[str, torch.Tensor]] = []
    handles = []

    def record(name: str):
        def hook(_module, _inputs, output: torch.Tensor) -> None:
            captured.append((name, output.detach()))

        return hook

    # The embedding dropout emits token+position vectors before any block has
    # run, which makes it the layer-0 baseline of the lens.
    handles.append(model.embedding_dropout.register_forward_hook(record("embeddings")))
    for block_number, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(record(f"block {block_number}")))

    try:
        model(tokens)
    finally:
        # Hooks stay attached to the module until removed, so leaving them
        # registered would make every later forward pass keep appending.
        for handle in handles:
            handle.remove()

    return captured


def lens_probabilities(
    model: TinyTransformer,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Decode a (B,T,C) residual stream into (B,T,V) next-token probabilities."""

    logits = model.language_model_head(model.final_norm(residual))
    return torch.softmax(logits, dim=-1)


def display_token(tokenizer: Tokenizer, token_id: int) -> str:
    """Render one token ID compactly, keeping whitespace visible."""

    if token_id == END_OF_TEXT:
        return "<end>"

    # vocab is consulted directly rather than tokenizer.decode because a single
    # BPE token can hold a partial UTF-8 sequence, and decode() also stops at
    # END_OF_TEXT. repr() then quotes the text so ' the' and 'the' stay distinct
    # and newlines print as \n instead of breaking the table.
    text = tokenizer.vocab[token_id].decode("utf-8", errors="replace")
    shown = repr(text)

    if len(shown) > TOKEN_COLUMN_WIDTH:
        shown = shown[: TOKEN_COLUMN_WIDTH - 1] + "…"

    return shown


def format_prediction(tokenizer: Tokenizer, token_id: int, probability: float) -> str:
    return f"{display_token(tokenizer, token_id):>{TOKEN_COLUMN_WIDTH}} {probability:5.1%}"


def print_lens_step(
    model: TinyTransformer,
    tokenizer: Tokenizer,
    tokens: torch.Tensor,
    sampled_token: int,
    top_k: int,
) -> None:
    """Print one table: what each layer predicted for the next token."""

    stream = capture_residual_stream(model, tokens)

    header = " ".join(
        f"{f'#{rank + 1}':>{TOKEN_COLUMN_WIDTH + 6}}" for rank in range(top_k)
    )
    print(f"  {'layer':<{LAYER_COLUMN_WIDTH}}{header}")

    sampled_progress = []
    for stage_number, (name, residual) in enumerate(stream):
        # [0, -1] selects the only batch item at the final position: the slot
        # whose prediction becomes the next generated token.
        probabilities = lens_probabilities(model, residual)[0, -1]
        top_probabilities, top_ids = probabilities.topk(top_k)

        cells = " ".join(
            format_prediction(tokenizer, token_id.item(), probability.item())
            for token_id, probability in zip(top_ids, top_probabilities)
        )
        # The last stage's decode is not an approximation: it is exactly what
        # TinyTransformer.forward returns, so its row is the model's real output.
        label = name + (" (real)" if stage_number == len(stream) - 1 else "")
        print(f"  {label:<{LAYER_COLUMN_WIDTH}}{cells}")

        # Where the eventually-sampled token sat in this layer's ranking.
        # argsort descending puts the most likely token first; the position of
        # sampled_token in that ordering is its rank.
        rank = (probabilities > probabilities[sampled_token]).sum().item() + 1
        sampled_progress.append(f"{name} #{rank} ({probabilities[sampled_token]:.1%})")

    print(
        f"  sampled {display_token(tokenizer, sampled_token):<{TOKEN_COLUMN_WIDTH}}"
        f"  {'  ->  '.join(sampled_progress)}"
    )


def run_demo(
    prompt: str,
    steps: int,
    temperature: float,
    top_k: int,
) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    torch.manual_seed(RANDOM_SEED)
    device = choose_device()
    tokenizer = Tokenizer.load(TOKENIZER_PATH)

    model = TinyTransformer().to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    model.eval()  # Dropout off, so the lens sees deterministic activations.

    starting_tokens = tokenizer.encode(prompt) or [END_OF_TEXT]
    tokens = torch.tensor([starting_tokens], dtype=torch.long, device=device)

    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"device:     {device}")
    print(f"prompt:     {prompt!r} -> {len(starting_tokens)} tokens")
    print("lens:       final_norm + language_model_head applied after every stage")
    print(
        "note:       the embeddings row mostly predicts the token already there.\n"
        "            The head shares its weights with the token embedding table,\n"
        "            so an untouched embedding decodes back to itself. Watch that\n"
        "            self-reference dissolve into a real prediction across blocks."
    )
    print()

    with torch.no_grad():
        for step in range(steps):
            context = tokens[:, -CONTEXT_LENGTH:]

            # The real sampling path, identical to generate.py: only the final
            # layer's logits choose the token. The lens is an observer.
            logits = model(context)[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            sampled_token = torch.multinomial(probabilities, num_samples=1)

            print(f"step {step + 1} | so far: {tokenizer.decode(tokens[0].tolist(), errors='replace')!r}")
            print_lens_step(
                model,
                tokenizer,
                context,
                sampled_token.item(),
                top_k,
            )
            print()

            tokens = torch.cat((tokens, sampled_token), dim=1)
            if sampled_token.item() == END_OF_TEXT:
                print("end-of-text sampled; stopping early")
                break

    print("completion:")
    print(tokenizer.decode(tokens[0].tolist(), errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=8, help="tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=5, help="predictions per layer")
    arguments = parser.parse_args()

    run_demo(
        arguments.prompt,
        arguments.steps,
        arguments.temperature,
        arguments.top_k,
    )
