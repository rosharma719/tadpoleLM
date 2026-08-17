"""Finish pretraining, personalize on notes, verify Hub, then destroy Vast.

Run this inside the Vast instance while pretraining is already in progress. It
never destroys the instance unless both stages report success and the expected
private Hugging Face files exist. On failure it stops the instance instead, so
GPU billing ends while the workspace remains available for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import HfApi


def process_is_running(pid: int) -> bool:
    """Return whether Linux still has the watched training process."""
    return Path(f"/proc/{pid}").exists()


def wait_for_process(pid: int, seconds: int) -> None:
    while process_is_running(pid):
        print(f"waiting for pretraining PID {pid}", flush=True)
        time.sleep(seconds)


def require_log_marker(path: Path, marker: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker not in text:
        raise RuntimeError(f"{path} does not contain success marker: {marker}")


def expected_pretrain_steps(checkpoint: dict) -> int:
    config = checkpoint["config"]
    blocks = (config["training_tokens"] - 1) // config["context_length"]
    return math.ceil(blocks / config["batch_size"])


def require_complete_checkpoint(path: Path, expected_steps: int) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != expected_steps:
        raise RuntimeError(
            f"{path} is at step {checkpoint['step']}, expected {expected_steps}"
        )


def require_hub_files(api: HfApi, repo: str, required: set[str]) -> None:
    files = set(api.list_repo_files(repo, repo_type="model"))
    missing = required - files
    if missing:
        raise RuntimeError(f"{repo} is missing Hub files: {sorted(missing)}")
    print(f"verified private Hub files in {repo}", flush=True)


def run_and_tee(command: list[str], log_path: Path) -> None:
    """Run fine-tuning in the foreground while saving a readable log."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"fine-tuning exited with status {return_code}")


def container_value(name: str) -> str:
    """Read a Vast control value, including values hidden from SSH shells."""
    if os.environ.get(name):
        return os.environ[name]
    environment = Path("/proc/1/environ").read_bytes().split(b"\0")
    prefix = f"{name}=".encode()
    for item in environment:
        if item.startswith(prefix):
            return item[len(prefix) :].decode()
    raise RuntimeError(f"Vast did not provide {name}")


def set_instance_state(state: str) -> None:
    """Stop this instance using its restricted per-instance Vast API key."""
    instance_id = container_value("CONTAINER_ID")
    api_key = container_value("CONTAINER_API_KEY")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        data=json.dumps({"state": state}).encode(),
        method="PUT",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode(), flush=True)


def destroy_instance() -> None:
    """Permanently destroy this instance after every artifact is verified."""
    instance_id = container_value("CONTAINER_ID")
    api_key = container_value("CONTAINER_API_KEY")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    print(f"destroying verified Vast instance {instance_id}", flush=True)
    with urllib.request.urlopen(request, timeout=30) as response:
        print(response.read().decode(), flush=True)


def finish(arguments: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    base_output = Path(arguments.base_output)
    pretrain_log = Path(arguments.pretrain_log)

    wait_for_process(arguments.pretrain_pid, arguments.poll_seconds)
    require_log_marker(
        pretrain_log,
        f"saved privately:   https://huggingface.co/{arguments.model_repo}",
    )
    base_latest = base_output / "latest.pt"
    checkpoint = torch.load(base_latest, map_location="cpu", weights_only=True)
    pretrain_steps = expected_pretrain_steps(checkpoint)
    require_complete_checkpoint(base_latest, pretrain_steps)
    require_hub_files(
        api,
        arguments.model_repo,
        {"best.pt", "latest.pt", "tokenizer.json", "config.json"},
    )
    print("pretraining completed and persisted; starting notes", flush=True)

    finetune_log = Path(arguments.finetune_log)
    command = [
        sys.executable,
        "-u",
        arguments.training_script,
        "finetune",
        "--model-repo",
        arguments.model_repo,
        "--notes",
        arguments.notes,
        "--base-output",
        arguments.base_output,
        "--output",
        arguments.notes_output,
        "--destination-repo",
        arguments.destination_repo,
        "--micro-batch-size",
        str(arguments.micro_batch_size),
    ]
    run_and_tee(command, finetune_log)
    require_log_marker(
        finetune_log,
        f"saved privately:   https://huggingface.co/{arguments.destination_repo}",
    )

    fine_config = json.loads(
        (Path(arguments.notes_output) / "finetune_config.json").read_text()
    )
    model_config = checkpoint["config"]
    fine_steps = math.ceil(
        fine_config["training_tokens"]
        / (fine_config["batch_size"] * model_config["context_length"])
    )
    require_complete_checkpoint(
        Path(arguments.notes_output) / "latest.pt", fine_steps
    )
    require_hub_files(
        api,
        arguments.destination_repo,
        {
            "best.pt",
            "latest.pt",
            "tokenizer.json",
            "config.json",
            "finetune_config.json",
        },
    )
    print("both stages are complete and safely stored on Hugging Face", flush=True)
    destroy_instance()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-pid", type=int, required=True)
    parser.add_argument("--pretrain-log", default="train.log")
    parser.add_argument("--training-script", default="cloud_train.py")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--destination-repo", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--base-output", default="cloud_runs/english_30m")
    parser.add_argument("--notes-output", default="cloud_runs/notes_30m")
    parser.add_argument("--finetune-log", default="finetune.log")
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=30)
    arguments = parser.parse_args()

    try:
        finish(arguments)
    except Exception as error:
        print(f"pipeline failed safely: {error}", flush=True)
        print("stopping instead of destroying so the workspace survives", flush=True)
        try:
            set_instance_state("stopped")
        except Exception as stop_error:
            print(f"automatic stop also failed: {stop_error}", flush=True)
        raise


if __name__ == "__main__":
    main()
