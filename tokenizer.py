"""A small byte-pair encoding tokenizer, written from scratch."""

import json
from collections import Counter
from pathlib import Path

from config import CORPUS_PATH, END_OF_TEXT, TOKENIZER_PATH, VOCAB_SIZE


# A byte has 8 bits, so it has 2**8 = 256 possible values: IDs 0 through 255.
# END_OF_TEXT occupies 256 and must not participate in merges, so BPE starts at 257.
FIRST_MERGE_ID = END_OF_TEXT + 1


def count_pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
    """Count every adjacent pair in the current token stream."""

    # tokens[1:] is tokens shifted one place left. zip aligns each token with
    # its right-hand neighbor; Counter then maps each pair to its frequency.
    # [10, 20, 10] becomes Counter({(10, 20): 1, (20, 10): 1}).
    return Counter(zip(tokens, tokens[1:]))


def merge_pair(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair with new_id."""

    merged = []
    position = 0

    while position < len(tokens):
        pair_starts_here = (
            position + 1 < len(tokens)
            and tokens[position] == pair[0]
            and tokens[position + 1] == pair[1]
        )

        if pair_starts_here:
            merged.append(new_id)
            position += 2
        else:
            merged.append(tokens[position])
            position += 1

    return merged


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        # __init__ runs after train/load has obtained the ordered merge list.
        self.merges = merges
        self.vocab = self.build_vocab()

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load a tokenizer from JSON without training it again."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in data["merges"]]
        # cls is Tokenizer when called as Tokenizer.load(...). Calling
        # cls(merges) constructs the instance, which then runs __init__ above.
        return cls(merges)

    @classmethod
    def train(cls, text: str, vocab_size: int = VOCAB_SIZE) -> "Tokenizer":
        """Learn BPE merges from text, choosing the most common pair each time."""

        if vocab_size < FIRST_MERGE_ID:
            raise ValueError(f"vocab_size must be at least {FIRST_MERGE_ID}")

        tokens = list(text.encode("utf-8"))
        merges = []
        merges_to_learn = vocab_size - FIRST_MERGE_ID

        for merge_number in range(merges_to_learn):
            pair_counts = count_pairs(tokens)
            if not pair_counts:
                break

            # Counter preserves insertion order, so ties are resolved by the
            # pair that appears first in the corpus.
            most_common_pair = max(pair_counts, key=pair_counts.get)
            new_id = FIRST_MERGE_ID + merge_number

            tokens = merge_pair(tokens, most_common_pair, new_id)
            merges.append(most_common_pair)

        # Training first discovers the constructor input; only now do we create
        # the Tokenizer instance and build its decoding vocabulary.
        return cls(merges)

    def build_vocab(self) -> dict[int, bytes]:
        """Build the byte sequence represented by every possible token ID."""

        vocab = {byte: bytes([byte]) for byte in range(256)}

        for merge_number, (left, right) in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + merge_number
            vocab[new_id] = vocab[left] + vocab[right]

        return vocab

    def encode(self, text: str, add_end: bool = False) -> list[int]:
        """Encode text as bytes, then replay the learned merges in order."""

        # Start with universal byte tokens, then replay merges in the exact
        # order learned; a later merge may refer to an earlier learned token.
        tokens = list(text.encode("utf-8"))

        for merge_number, pair in enumerate(self.merges):
            new_id = FIRST_MERGE_ID + merge_number
            tokens = merge_pair(tokens, pair, new_id)

        if add_end:
            tokens.append(END_OF_TEXT)

        return tokens

    def decode(self, tokens: list[int], errors: str = "strict") -> str:
        """Expand token IDs back to their bytes, stopping at end-of-text."""

        pieces = []

        for token in tokens:
            if token == END_OF_TEXT:
                break
            if token not in self.vocab:
                raise ValueError(f"invalid token: {token}")
            pieces.append(self.vocab[token])

        return b"".join(pieces).decode("utf-8", errors=errors)

    def save(self, path: str | Path) -> None:
        """Save the learned tokenizer as readable JSON."""

        data = {
            "end_of_text": END_OF_TEXT,
            "merges": self.merges,
            "vocab": {
                str(token_id): list(token_bytes)
                for token_id, token_bytes in self.vocab.items()
            },
        }

        Path(path).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    corpus = Path(CORPUS_PATH).read_text(encoding="utf-8")
    tokenizer_path = Path(TOKENIZER_PATH)

    if tokenizer_path.exists():
        tokenizer = Tokenizer.load(tokenizer_path)
        print("loaded:          tokenizer.json")
    else:
        print(f"training a {VOCAB_SIZE}-token vocabulary...")
        tokenizer = Tokenizer.train(corpus)
        tokenizer.save(tokenizer_path)
        print("saved:           tokenizer.json")

    raw_tokens = list(corpus.encode("utf-8"))
    bpe_tokens = tokenizer.encode(corpus)

    print(f"raw byte tokens: {len(raw_tokens):,}")
    print(f"BPE tokens:      {len(bpe_tokens):,}")
    print(f"compression:     {len(raw_tokens) / len(bpe_tokens):.2f} bytes/token")

    assert tokenizer.decode(bpe_tokens) == corpus
    print("round trip:      passed")
