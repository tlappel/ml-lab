"""Generate text from a trained checkpoint.

    python gpt/sample.py
    python gpt/sample.py --prompt "ROMEO:"
    python gpt/sample.py --prompt "KING RICHARD III:" --tokens 500 --temperature 0.6
    python gpt/sample.py --samples 3 --temperature 1.2

This loads the BEST checkpoint - the one saved at the lowest validation loss -
not the final state of training. If your run overfitted (train loss falling
while val loss rose), this model is meaningfully better than the sample
train.py printed at the end.

What this does is the entire inference loop of every language model: predict a
distribution over the next token, sample one, append it, feed it all back in.
Nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPT

CKPT = Path(__file__).parent / "out" / "ckpt.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="\n", help="text to continue from")
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="<1 conservative and repetitive, >1 wilder and less coherent",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="only sample from the k likeliest characters; 0 disables",
    )
    ap.add_argument("--ckpt", default=str(CKPT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = Path(args.ckpt)
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} - run train.py first")

    # weights_only=False because the checkpoint stores a GPTConfig dataclass
    # alongside the tensors. Only ever do this with files you produced.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()  # disables dropout - important, and easy to forget

    # Rebuild the tokenizer from the exact character set training used. The
    # mapping is positional, so a different vocabulary order would silently
    # produce garbage rather than an error.
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.chars = ckpt["chars"]
    tokenizer.vocab_size = len(ckpt["chars"])
    tokenizer._stoi = {ch: i for i, ch in enumerate(ckpt["chars"])}
    tokenizer._itos = dict(enumerate(ckpt["chars"]))

    print(f"checkpoint  iter {ckpt['iter']}, val loss {ckpt['val_loss']:.4f}")
    print(f"device      {device}")
    print(f"prompt      {args.prompt!r}")
    print(f"temperature {args.temperature}   top_k {args.top_k or 'off'}")
    print("-" * 70)

    unknown = sorted(set(args.prompt) - set(ckpt["chars"]))
    if unknown:
        raise SystemExit(
            f"prompt contains characters the model has never seen: {unknown}\n"
            f"this is a character-level model - its whole vocabulary is "
            f"{len(ckpt['chars'])} symbols from the training corpus."
        )

    idx = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )

    for n in range(args.samples):
        out = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )
        print(tokenizer.decode(out[0].tolist()))
        if n < args.samples - 1:
            print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
