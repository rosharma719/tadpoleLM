"""Create a roughly 3x training corpus by paraphrasing the notes with OpenAI."""

import argparse
import json
import os
from pathlib import Path

from config import (
    AUGMENTED_CORPUS_PATH,
    CORPUS_PATH,
    PARAPHRASE_CHUNK_CHARACTERS,
    PARAPHRASE_MAX_OUTPUT_TOKENS,
    PARAPHRASE_MODEL,
    PARAPHRASE_PROGRESS_PATH,
    PARAPHRASE_VARIANTS,
)


INSTRUCTIONS = """Paraphrase the supplied personal notes.

Rules:
- Preserve every idea, factual claim, uncertainty, and first-person point of view.
- Preserve the Markdown bullet hierarchy.
- Change the wording and sentence structure substantially.
- Keep roughly the same amount of detail and approximately the same length.
- Do not answer the notes, add facts, censor them, summarize them, or add a heading.
- Return only the paraphrased notes.
"""


def split_corpus(text: str, target_characters: int) -> list[str]:
    """Split near a target size, but never through a top-level ``- `` note."""

    chunks = []
    current_lines = []
    current_length = 0

    for line in text.splitlines(keepends=True):
        starts_top_level_note = line.startswith("- ")

        if starts_top_level_note and current_length >= target_characters:
            chunks.append("".join(current_lines).strip())
            current_lines = []
            current_length = 0

        current_lines.append(line)
        current_length += len(line)

    if current_lines:
        chunks.append("".join(current_lines).strip())

    return [chunk for chunk in chunks if chunk]


def load_progress(path: Path) -> dict[tuple[int, int], str]:
    """Load successful earlier responses so a stopped run can continue."""

    completed = {}
    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            key = (record["chunk"], record["variant"])
            completed[key] = record["text"]

    return completed


def load_api_key(path: Path = Path(".env")) -> str | None:
    """Read OPENAI_API_KEY from the environment or a small local .env file."""

    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            return value.strip().strip("\"'")

    return None


def request_paraphrase(client, text: str, variant: int) -> str:
    """Send one chunk to the Responses API and return its rewritten text."""

    response = client.responses.create(
        model=PARAPHRASE_MODEL,
        instructions=INSTRUCTIONS,
        input=f"Rewrite variant {variant}:\n\n{text}",
        max_output_tokens=PARAPHRASE_MAX_OUTPUT_TOKENS,
        reasoning={"effort": "minimal"},
        store=False,
    )
    paraphrase = response.output_text.strip()

    if not paraphrase:
        raise RuntimeError("The API returned an empty paraphrase.")

    return paraphrase


def save_augmented_corpus(
    chunks: list[str],
    paraphrases: dict[tuple[int, int], str],
    output_path: Path,
) -> None:
    """Interleave each original chunk with its rewrites and save one corpus."""

    sections = []
    for chunk_index, original in enumerate(chunks):
        sections.append(original)
        for variant in range(1, PARAPHRASE_VARIANTS + 1):
            sections.append(paraphrases[(chunk_index, variant)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def main(run: bool) -> None:
    source_path = Path(CORPUS_PATH)
    progress_path = Path(PARAPHRASE_PROGRESS_PATH)
    output_path = Path(AUGMENTED_CORPUS_PATH)

    source = source_path.read_text(encoding="utf-8")
    chunks = split_corpus(source, PARAPHRASE_CHUNK_CHARACTERS)
    total_calls = len(chunks) * PARAPHRASE_VARIANTS
    completed = load_progress(progress_path)

    print(f"source characters: {len(source):,}")
    print(f"chunks:            {len(chunks):,}")
    print(f"API calls:         {total_calls:,}")
    print(f"already completed: {len(completed):,}")
    print(f"model:             {PARAPHRASE_MODEL}")

    if not run:
        print("preview only; run `python augment.py --run` to call the API")
        return

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in .env before using --run.")

    # This is the only extra package this script needs. Importing it here means
    # the free preview above still works before `openai` has been installed.
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    with progress_path.open("a", encoding="utf-8") as progress_file:
        for chunk_index, chunk in enumerate(chunks):
            for variant in range(1, PARAPHRASE_VARIANTS + 1):
                key = (chunk_index, variant)
                if key in completed:
                    continue

                print(
                    f"request {len(completed) + 1}/{total_calls} "
                    f"(chunk {chunk_index + 1}/{len(chunks)}, variant {variant})"
                )
                paraphrase = request_paraphrase(client, chunk, variant)
                completed[key] = paraphrase

                record = {
                    "chunk": chunk_index,
                    "variant": variant,
                    "text": paraphrase,
                }
                progress_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress_file.flush()

    save_augmented_corpus(chunks, completed, output_path)
    output_characters = output_path.stat().st_size
    print(f"saved:             {output_path}")
    print(f"output bytes:      {output_characters:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="make paid API calls; without this flag the script only previews",
    )
    arguments = parser.parse_args()
    main(arguments.run)
