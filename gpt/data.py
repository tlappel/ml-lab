"""Corpus loading and a character-level tokenizer.

Character-level is the right choice for learning. Real models use subword
tokenizers (BPE), which are a genuinely separate topic with their own
machinery. Characters let you skip all of that and still see every other part
of the system work honestly.

The cost: a character carries less meaning than a word piece, so a small model
has to spend capacity learning spelling before it can learn anything else.
That's why the first samples look like gibberish with the right *shape* —
line lengths, capitalization, punctuation rhythm — before they look like words.
Watching that progression is half the point.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch

TINY_SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

DATA_DIR = Path(__file__).parent / "data"


def load_text(name: str = "input.txt", url: str = TINY_SHAKESPEARE) -> str:
    """Return the corpus, downloading it once if needed."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if not path.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """Maps characters to integers and back.

    A tokenizer is nothing more than an agreed-upon numbering of the symbols.
    The model never sees text — it sees integers, which it immediately uses to
    look up rows in an embedding table. Everything downstream is vectors.
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self._stoi = {ch: i for i, ch in enumerate(self.chars)}
        self._itos = dict(enumerate(self.chars))

    def encode(self, s: str) -> list[int]:
        return [self._stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)


def train_val_split(
    text: str, tokenizer: CharTokenizer, val_frac: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the whole corpus once, then split by position.

    Split by position rather than randomly because the data is a single
    continuous stream — a random split would put text from the middle of a
    sentence in train and its continuation in val, which leaks.
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cut = int(len(data) * (1 - val_frac))
    return data[:cut], data[cut:]


def get_batch(
    data: torch.Tensor, batch_size: int, block_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (context, target) pairs.

    This is the trick that makes language modelling so data-efficient. For a
    context of length `block_size`, the targets are the same sequence shifted
    one position left. So a single sequence of 256 characters produces 256
    separate prediction problems — predict char 1 from char 0, char 2 from
    chars 0-1, and so on — all trained in parallel in one forward pass.

    Shapes: both come back (batch_size, block_size). Call them (B, T)
    everywhere; that naming carries through the whole model.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    if device == "cuda":
        # pin_memory + non_blocking lets the copy overlap with compute.
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
