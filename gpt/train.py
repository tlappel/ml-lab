"""Train the small GPT. This file is complete — it runs as soon as model.py does.

    python gpt/train.py

Roughly 10.8M parameters at the default config. Trains in a few minutes on a
12GB card and uses well under 2GB, so there's a lot of headroom to play with:
push n_layer, n_embd, or block_size up and watch both the loss and the memory
move. Learning where the wall is, by hitting it, is worth doing deliberately.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from data import CharTokenizer, get_batch, load_text, train_val_split
from model import GPT, GPTConfig

# ---------------------------------------------------------------- config

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 250
eval_iters = 200
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
weight_decay = 0.1
grad_clip = 1.0

# torch.compile gives a solid speedup on Linux but needs Triton, which has no
# Windows build. Leave this off here; turn it on if you ever move to WSL.
use_compile = False

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = Path(__file__).parent / "out"

torch.manual_seed(1337)
# Allow TF32 for the fp32 ops that remain — free accuracy-for-speed trade
# on anything Ampere or newer.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_lr(it: int) -> float:
    """Linear warmup, then cosine decay.

    Warmup exists because the first steps are taken from random weights, where
    gradients are large and meaningless. Going full speed there can wreck the
    run in a dozen steps. Cosine decay at the end lets the model settle into a
    minimum instead of bouncing around it.
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, splits: dict[str, torch.Tensor]) -> dict[str, float]:
    """Average loss over several batches, in eval mode.

    Single-batch loss is far too noisy to judge anything by. And model.eval()
    matters: it disables dropout, which would otherwise make the model look
    worse than it is.
    """
    out = {}
    model.eval()
    for split, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    text = load_text()
    tokenizer = CharTokenizer(text)
    train_data, val_data = train_val_split(text, tokenizer)
    splits = {"train": train_data, "val": val_data}

    print(f"corpus      {len(text):,} chars")
    print(f"vocab       {tokenizer.vocab_size} symbols")
    print(f"train/val   {len(train_data):,} / {len(val_data):,} tokens")

    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model = GPT(config).to(device)
    print(f"params      {model.num_params()/1e6:.2f}M")
    print(f"device      {device}")
    print()

    # A useful sanity number: at initialisation the model knows nothing, so it
    # should predict a uniform distribution over the vocabulary. That gives a
    # loss of ln(vocab_size) — about 4.17 for 65 characters. If your first
    # loss is far from that, something is wrong with the init, not the
    # training. This is the cheapest bug-catch in the whole file.
    print(f"expected initial loss ~{math.log(tokenizer.vocab_size):.2f}")

    if use_compile:
        model = torch.compile(model)

    # Weight decay on matmul weights only. Decaying biases, layernorm gains,
    # and embeddings hurts — those aren't the parameters overfitting is made of.
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )

    out_dir.mkdir(exist_ok=True)
    best_val = float("inf")
    t0 = time.perf_counter()

    for it in range(max_iters):
        lr = get_lr(it)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, splits)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            elapsed = time.perf_counter() - t0
            print(
                f"iter {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}  "
                f"lr {lr:.2e}  {mem:.2f} GiB  {elapsed:.0f}s"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": config,
                        "chars": tokenizer.chars,
                        "iter": it,
                        "val_loss": best_val,
                    },
                    out_dir / "ckpt.pt",
                )

        x, y = get_batch(train_data, batch_size, block_size, device)

        # bfloat16 autocast. bf16 rather than fp16 because it keeps fp32's
        # exponent range — so no gradient scaler, no overflow babysitting.
        # It's the reason mixed precision stopped being fiddly.
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip the global gradient norm. A single bad batch can otherwise
        # produce an enormous update that undoes hours of training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    print(f"\ndone in {time.perf_counter() - t0:.0f}s — best val {best_val:.4f}\n")
    print("-" * 60)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=1000, temperature=0.8, top_k=40)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
