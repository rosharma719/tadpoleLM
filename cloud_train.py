# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "huggingface-hub>=0.34,<2",
#   "numpy>=2.0,<3",
#   "pyarrow>=21,<27",
#   "tokenizers>=0.21,<1",
#   "torch>=2.7,<3",
# ]
# ///
"""Pretrain the 30M Tadpole model on English, then generate from it.

This file is self-contained so Hugging Face Jobs can upload and run it. It is
separate from the handwritten teaching implementation in model.py/train.py.

Local smoke test:
    uv run cloud_train.py pretrain --smoke

Cloud training (the exact command is printed by ``python cloud_train.py job``):
    hf jobs uv run ... cloud_train.py pretrain --model-repo USER/REPO \
        --support gpt.py lm_utils.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch import nn

from gpt import Transformer, TransformerConfig
from lm_utils import (
    choose_device,
    cosine_learning_rate,
    generate_tokens,
    next_token_loss,
)


# Download the second shard only if the first cannot provide all requested tokens.
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_FILES = (
    "sample/10BT/000_00000.parquet",
    "sample/10BT/000_00001.parquet",
)
END_OF_TEXT = "<|endoftext|>"


@dataclass
class Config(TransformerConfig):
    # Architecture: an approximately 30M-parameter GPT.
    vocab_size: int = 8192       # Larger English BPE vocabulary.
    context_length: int = 512    # Twice the previous attention window.
    embedding_size: int = 512    # Residual-stream width C.
    number_of_heads: int = 8     # Eight heads of 64 dimensions each.
    number_of_blocks: int = 8    # Eight attention + feed-forward blocks.
    feed_forward_multiplier: int = 4
    dropout: float = 0.1
    weight_init_std: float = 0.02

    # Dataset. Validation is held out before the training portion.
    training_tokens: int = 600_000_000
    validation_tokens: int = 5_000_000
    tokenizer_documents: int = 100_000

    # Optimization. 256 * 512 = 131,072 tokens are processed per update.
    batch_size: int = 256
    learning_rate: float = 6e-4
    minimum_learning_rate: float = 6e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    evaluation_interval: int = 100
    evaluation_batches: int = 20
    upload_interval: int = 500     # Persist to Hub without stalling every eval.
    sample_tokens: int = 50
    seed: int = 42


def train_tokenizer(documents, path: Path, config: Config) -> Tokenizer:
    """Learn a byte-level BPE vocabulary from an iterator of English strings."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        documents,
        trainer=trainer,
        length=config.tokenizer_documents,
    )
    tokenizer.save(str(path))
    # A tiny smoke corpus can run out of distinct pairs before reaching 4,096.
    # FineWeb is diverse enough that the real run reaches the requested size.
    print(f"tokenizer vocab:   {tokenizer.get_vocab_size():,}")
    return tokenizer


def fineweb_documents(path: Path, limit: int | None = None):
    """Yield text from a local Parquet shard without creating an Arrow copy."""
    produced = 0
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=256, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if limit is not None and produced >= limit:
                return
            if text:
                yield text
                produced += 1


def downloaded_fineweb_documents(directory: Path):
    """Download shards lazily and yield their documents in order."""
    for filename in FINEWEB_FILES:
        print(f"using shard:        {filename}", flush=True)
        path = Path(
            hf_hub_download(
                repo_id=FINEWEB_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=directory,
            )
        )
        yield from fineweb_documents(path)


def smoke_documents(limit: int | None = None):
    """Small built-in corpus used only to verify the pipeline without a download."""
    examples = [
        "A small language model predicts the next token in a sequence.",
        "Attention lets each token gather information from earlier tokens.",
        "The optimizer changes weights to reduce cross entropy loss.",
        "A byte level tokenizer can represent every Unicode string.",
        "Training uses many sequences at once, while generation is autoregressive.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    produced = 0
    for _ in range(500):
        for example in examples:
            if limit is not None and produced >= limit:
                return
            yield example
            produced += 1


def write_token_data(
    tokenizer: Tokenizer,
    documents,
    directory: Path,
    config: Config,
) -> tuple[Path, Path]:
    """Encode whole documents and write validation/train int16 token streams."""
    validation_path = directory / "validation.bin"
    training_path = directory / "training.bin"
    eot = tokenizer.token_to_id(END_OF_TEXT)
    validation_count = 0
    training_count = 0
    last_report = 0

    with validation_path.open("wb") as validation_file, training_path.open(
        "wb"
    ) as training_file:
        for text in documents:
            ids = tokenizer.encode(text).ids + [eot]
            if validation_count < config.validation_tokens:
                needed = config.validation_tokens - validation_count
                chunk = ids[:needed]
                np.asarray(chunk, dtype=np.int16).tofile(validation_file)
                validation_count += len(chunk)
                # Never put the remainder of a validation document in training.
                # This avoids measuring the model on the first half of a document
                # after it trained on that document's second half.
                continue

            if ids and training_count < config.training_tokens:
                needed = config.training_tokens - training_count
                chunk = ids[:needed]
                np.asarray(chunk, dtype=np.int16).tofile(training_file)
                training_count += len(chunk)

            total = validation_count + training_count
            if total - last_report >= 10_000_000:
                print(f"encoded tokens:    {total:,}", flush=True)
                last_report = total
            if (
                validation_count >= config.validation_tokens
                and training_count >= config.training_tokens
            ):
                break

    if validation_count < config.validation_tokens:
        raise RuntimeError(f"only found {validation_count:,} validation tokens")
    if training_count < config.training_tokens:
        raise RuntimeError(f"only found {training_count:,} training tokens")
    return training_path, validation_path


def token_tensor(path: Path) -> torch.Tensor:
    """Map the binary file without copying all of it into a second CPU array."""
    # Copy-on-write gives PyTorch a writable view without copying the 500 MB
    # corpus into RAM. We still only read it.
    # int16 is enough for our 8,192 token IDs and, unlike torch.uint16,
    # supports advanced indexing in PyTorch on both macOS and Linux.
    mapped = np.memmap(path, dtype=np.int16, mode="c")
    return torch.from_numpy(mapped)


def make_batch(
    data: torch.Tensor,
    block_numbers: torch.Tensor,
    context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn B block numbers into inputs/targets shaped (B, T)."""
    offsets = torch.arange(context + 1)
    indexes = block_numbers[:, None] * context + offsets[None, :]
    sequences = data[indexes].long()
    return (
        sequences[:, :-1].to(device, non_blocking=True),
        sequences[:, 1:].to(device, non_blocking=True),
    )


@torch.inference_mode()
def evaluate(
    model: Transformer,
    data: torch.Tensor,
    config: Config,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool,
) -> float:
    model.eval()
    blocks = min(
        (len(data) - 1) // config.context_length,
        config.evaluation_batches * batch_size,
    )
    losses = []
    for start in range(0, blocks, batch_size):
        numbers = torch.arange(start, min(start + batch_size, blocks))
        x, y = make_batch(data, numbers, config.context_length, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=mixed_precision,
        ):
            losses.append(next_token_loss(model(x), y).item())
    model.train()
    return sum(losses) / len(losses)


def sample(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    count: int,
    temperature: float,
    device: torch.device,
) -> str:
    was_training = model.training
    model.eval()
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id(END_OF_TEXT)]
    generated = generate_tokens(
        model,
        ids,
        count,
        temperature,
        device,
        end_of_text=tokenizer.token_to_id(END_OF_TEXT),
    )
    model.train(was_training)
    return tokenizer.decode(generated, skip_special_tokens=False)


def save_checkpoint(
    path: Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_loss: float,
    config: Config,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "step": step,
        "best_validation_loss": best_loss,
        "config": asdict(config),
    }
    # latest.pt needs AdamW's state for resuming; best.pt is inference-only and
    # therefore about one third the size.
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def upload_artifacts(
    api: HfApi | None,
    repo: str | None,
    paths: list[Path],
    message: str,
) -> None:
    if api is None or repo is None:
        return
    for path in paths:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo,
            repo_type="model",
            commit_message=message,
        )


def pretrain(arguments: argparse.Namespace) -> None:
    config = Config()
    if arguments.smoke:
        # Tiny values exercise every stage; architecture stays approximately 30M.
        config.training_tokens = 16_384
        config.validation_tokens = 4_096
        config.tokenizer_documents = 1_000
        config.batch_size = 8
        config.evaluation_batches = 2
        config.evaluation_interval = 2
        config.warmup_steps = 1

    micro_batch_size = arguments.micro_batch_size or config.batch_size
    if micro_batch_size < 1 or micro_batch_size > config.batch_size:
        raise ValueError(
            f"--micro-batch-size must be between 1 and {config.batch_size}"
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(choose_device(cuda_first=True))
    mixed_precision = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    config_path = output / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")

    print(f"device:            {device}")
    print(f"output:            {output}")
    source_directory = None
    if arguments.smoke:
        source = smoke_documents
    else:
        source_directory = output / "source"
        print("downloading first FineWeb-Edu shard", flush=True)
        source_path = Path(
            hf_hub_download(
                repo_id=FINEWEB_REPO,
                filename=FINEWEB_FILES[0],
                repo_type="dataset",
                local_dir=source_directory,
            )
        )
        tokenizer_source = lambda limit=None: fineweb_documents(
            source_path, limit
        )
        source = lambda limit=None: downloaded_fineweb_documents(
            source_directory
        )

    print("training tokenizer", flush=True)
    if arguments.smoke:
        tokenizer_source = source
    tokenizer = train_tokenizer(
        tokenizer_source(config.tokenizer_documents), tokenizer_path, config
    )
    print("encoding corpus", flush=True)
    training_path, validation_path = write_token_data(
        tokenizer, source(), output, config
    )
    if source_directory is not None and not arguments.keep_source:
        shutil.rmtree(source_directory)
        print("deleted downloaded source shard", flush=True)
    training_data = token_tensor(training_path)
    validation_data = token_tensor(validation_path)

    model = Transformer(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (0.9, 0.95),
    }
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    number_of_blocks = (len(training_data) - 1) // config.context_length
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(number_of_blocks, generator=generator)
    total_steps = (
        number_of_blocks + config.batch_size - 1
    ) // config.batch_size

    api = None
    if arguments.model_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required when --model-repo is set")
        api = HfApi(token=token)
        api.create_repo(
            arguments.model_repo,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        upload_artifacts(
            api,
            arguments.model_repo,
            [tokenizer_path, config_path],
            "Add tokenizer and training config",
        )

    print(f"training tokens:   {len(training_data):,}")
    print(f"validation tokens: {len(validation_data):,}")
    print(f"parameters:        {parameter_count:,}")
    print(f"steps:             {total_steps:,}")
    print(
        f"tokens per update: {config.batch_size * config.context_length:,}"
    )
    print(
        f"micro batch:       {micro_batch_size} sequences"
        f" ({math.ceil(config.batch_size / micro_batch_size)} accumulations)",
        flush=True,
    )

    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    best_loss = float("inf")
    start_step = 0
    if arguments.resume:
        if not arguments.model_repo:
            raise ValueError("--resume requires --model-repo")
        resume_path = Path(
            hf_hub_download(
                arguments.model_repo,
                "latest.pt",
                token=os.environ["HF_TOKEN"],
            )
        )
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=True
        )
        if checkpoint["config"] != asdict(config):
            raise RuntimeError("saved checkpoint uses a different config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_loss = checkpoint["best_validation_loss"]
        print(f"resuming at step:  {start_step:,}")

    started = time.monotonic()
    model.train()

    for step in range(start_step, total_steps):
        lr = cosine_learning_rate(
            step,
            config.learning_rate,
            config.minimum_learning_rate,
            config.warmup_steps,
            total_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        start = step * config.batch_size
        step_numbers = order[start : start + config.batch_size]
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for micro_start in range(0, len(step_numbers), micro_batch_size):
            numbers = step_numbers[
                micro_start : micro_start + micro_batch_size
            ]
            x, y = make_batch(
                training_data, numbers, config.context_length, device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=mixed_precision,
            ):
                micro_loss = next_token_loss(model(x), y)
                fraction = len(numbers) / len(step_numbers)
                scaled_loss = micro_loss * fraction
            scaled_loss.backward()
            training_loss += micro_loss.item() * fraction
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        should_evaluate = (
            step == 0
            or (step + 1) % config.evaluation_interval == 0
            or step + 1 == total_steps
        )
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_data,
                config,
                micro_batch_size,
                device,
                mixed_precision,
            )
            improved = validation_loss < best_loss
            best_loss = min(best_loss, validation_loss)
            save_checkpoint(
                latest_path, model, optimizer, step + 1, best_loss, config
            )
            uploads = [latest_path]
            if improved:
                save_checkpoint(
                    best_path, model, None, step + 1, best_loss, config
                )
            should_upload = (
                step == 0
                or (step + 1) % config.upload_interval == 0
                or step + 1 == total_steps
            )
            if should_upload:
                if best_path.exists():
                    uploads.append(best_path)
                upload_artifacts(
                    api,
                    arguments.model_repo,
                    uploads,
                    f"Checkpoint after step {step + 1}",
                )
            elapsed = (time.monotonic() - started) / 60
            print(
                f"step {step + 1:4d}/{total_steps} | "
                f"train {training_loss:.4f} | validation {validation_loss:.4f} | "
                f"lr {lr:.2e} | {elapsed:.1f} min"
                f"{' | saved best' if improved else ''}",
                flush=True,
            )
            print(sample(model, tokenizer, "The meaning of", config.sample_tokens,
                         0.8, device), flush=True)

    print(f"best validation:   {best_loss:.4f}")
    print(f"saved locally:     {best_path}")
    if arguments.model_repo:
        print(f"saved privately:   https://huggingface.co/{arguments.model_repo}")


def load_saved_model(
    model_repo: str | None, directory: Path
) -> tuple[Transformer, Tokenizer]:
    if model_repo:
        token = os.environ.get("HF_TOKEN")
        checkpoint_path = Path(
            hf_hub_download(model_repo, "best.pt", token=token)
        )
        tokenizer_path = Path(
            hf_hub_download(model_repo, "tokenizer.json", token=token)
        )
    else:
        checkpoint_path = directory / "best.pt"
        tokenizer_path = directory / "tokenizer.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    model = Transformer(config)
    model.load_state_dict(checkpoint["model"])
    return model, Tokenizer.from_file(str(tokenizer_path))


def generate(arguments: argparse.Namespace) -> None:
    device = torch.device(choose_device(cuda_first=True))
    model, tokenizer = load_saved_model(
        arguments.model_repo, Path(arguments.output)
    )
    model.to(device)
    print(
        sample(
            model,
            tokenizer,
            arguments.prompt,
            arguments.tokens,
            arguments.temperature,
            device,
        )
    )


def print_job_command(arguments: argparse.Namespace) -> None:
    repo = arguments.model_repo or "YOUR_USERNAME/tadpole-english-30m"
    print(
        "hf jobs uv run --detach --flavor a100-large --timeout 3h "
        "-s HF_TOKEN cloud_train.py pretrain "
        f"--model-repo {repo} --support gpt.py lm_utils.py"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("pretrain")
    train_parser.add_argument("--output", default="cloud_runs/english_30m")
    train_parser.add_argument("--model-repo")
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument(
        "--keep-source",
        action="store_true",
        help="keep downloaded Parquet shards after tokenization",
    )
    train_parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="sequences held in memory at once; gradients accumulate to 256",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from latest.pt in --model-repo after rebuilding token data",
    )
    # The Jobs CLI bundles local file arguments beside this script. The program
    # itself does not otherwise need these values.
    train_parser.add_argument("--support", nargs="*", help=argparse.SUPPRESS)

    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("prompt", nargs="?", default="The meaning of")
    generate_parser.add_argument("--tokens", type=int, default=200)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    generate_parser.add_argument("--output", default="cloud_runs/english_30m")
    generate_parser.add_argument("--model-repo")

    job_parser = commands.add_parser("job")
    job_parser.add_argument("--model-repo")

    arguments = parser.parse_args()
    if arguments.command == "pretrain":
        pretrain(arguments)
    elif arguments.command == "generate":
        generate(arguments)
    else:
        print_job_command(arguments)


if __name__ == "__main__":
    main()
