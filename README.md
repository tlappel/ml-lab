# ml-lab

A workbench for learning ML on an RTX PRO 3000 Blackwell (Laptop, 12GB).

## Hardware notes — read these before debugging anything

**The card**

| | |
|---|---|
| GPU | NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU |
| VRAM | 12227 MiB (~12 GB) — **this is the real constraint** |
| Driver | 596.53 / CUDA 13.2 |
| Arch | Blackwell, `sm_120` |
| System RAM | ~72 GB |

**Task Manager lies.** It reports "shared GPU memory" (~36 GB) as if it were
usable VRAM and shows a "total" around 48 GB. That shared pool is system RAM the
driver is *allowed* to borrow over PCIe. When a model spills into it, Windows'
sysmem fallback doesn't crash — it crawls. Plan against 12 GB. Always.

**Blackwell needs CUDA 12.8+.** The default PyTorch wheels don't build for
`sm_120`. Plain `pip install torch` looks like it works, then the first `.cuda()`
call fails with `no kernel image is available for execution on the device` —
an error that says nothing about architecture. Install like this:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Same class of trap downstream: anything pinned to a pre-Blackwell CUDA
(old llama.cpp builds, old bitsandbytes) will fail the same confusing way.
Rule for this machine: newest stack, always.

**Never benchmark on battery.** Measured with `bench.py`:

| Condition | bf16 matmul |
|---|---|
| On battery, Balanced | 53.8 TFLOPS |
| Plugged in, Best Performance | **90.3 TFLOPS** |

A 68% swing from one cable. Before trusting any number: plugged in, and
Settings → System → Power & battery → Power mode → Best Performance. That
slider is a different control from the power *plan*, and it exists even when
IT has removed every plan but Balanced.

NVIDIA Control Panel has no "Power management mode" on this card — that's
expected, not missing. Dynamic Boost arbitrates the CPU/GPU power budget at
runtime, so there's no static preference to set.

## The shape of this machine

Compute is abundant; memory is scarce. 90 TFLOPS with 12 GB means the skill
this hardware teaches is **fitting things into memory**:

- QLoRA over full fine-tuning — 4-bit base, small trainable adapters
- Quantized inference always — Q4/Q5 GGUF or AWQ, never fp16 you don't need
- Gradient checkpointing on — spends surplus compute to buy memory

One calibration: **token generation speed won't track 90 TFLOPS.** Generating
text is memory-bandwidth bound — the card reads the whole model per token.
That number shows up in batch work: training, embeddings, prompt processing.

## Setup

```powershell
uv python install 3.12
uv venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install numpy
```

3.12 rather than 3.13 — the ML wheel ecosystem runs a version behind.

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python bench.py
```

## Contents

| Path | What |
|---|---|
| `bench.py` | Matmul benchmark with VRAM + power reporting |
| `gpt/data.py` | Downloads a corpus, builds a character tokenizer |
| `gpt/model.py` | **Incomplete on purpose.** The transformer, with TODOs. |
| `gpt/train.py` | Training loop, eval, sampling, checkpoints |

`model.py` is where the learning is. It ships with the scaffolding and
comments but the actual mechanism — attention, the feed-forward block, the
residual stream — left as TODOs. Filling those in is the exercise. `train.py`
is complete and will run the moment the model does.

bf16   ~100 TFLOPS     <- use this, always
fp16    ~65            <- 35% slower on this silicon. avoid.
tf32    ~44            <- free 2.4x over fp32, one line
fp32    ~18.5
VRAM    11.9 GiB       <- the actual constraint
