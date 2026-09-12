"""Matmul benchmark with memory, power, and thermal-confound controls.

    python bench.py                  # default order, 5s cooldown between runs
    python bench.py --reverse        # reverse order - isolates thermal drift
    python bench.py --cooldown 20    # longer settle between precisions
    python bench.py --repeat 3       # each precision 3x, reports best

Run plugged in, Power mode on Best Performance. On battery this card turns in
roughly 60% of its real throughput.
"""

from __future__ import annotations

import argparse
import subprocess
import time

import torch


def device_report() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. If torch is installed it's probably the wrong "
            "build - Blackwell needs cu128:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )
    p = torch.cuda.get_device_properties(0)
    print(f"device      {p.name}")
    print(f"capability  sm_{p.major}{p.minor}")
    print(f"vram        {p.total_memory / 1024**3:.1f} GiB")
    print(f"sms         {p.multi_processor_count}")
    print(f"torch       {torch.__version__}  (cuda {torch.version.cuda})")
    print()


def gpu_telemetry() -> str:
    """Power, clock and temperature straight from the driver.

    Every field is parsed independently and tolerates "[N/A]". Many laptop
    GPUs - including this one - simply do not expose power.draw through NVML,
    and an all-or-nothing parse threw away the clock and temperature readings
    that WERE available. Partial telemetry beats none.
    """
    fields = ("power.draw", "power.limit", "clocks.sm", "temperature.gpu")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "nvidia-smi not found"

    if out.returncode != 0:
        return f"nvidia-smi failed: {out.stderr.strip()[:60]}"

    vals = [v.strip() for v in out.stdout.strip().split(",")]
    if len(vals) != len(fields):
        return f"unparsed: {out.stdout.strip()[:60]}"

    def num(v):
        try:
            return float(v)
        except ValueError:
            return None  # "[N/A]" / "[Not Supported]"

    draw, limit, clk, temp = (num(v) for v in vals)

    parts = []
    if draw is not None and limit is not None:
        parts.append(f"{draw:3.0f}/{limit:3.0f}W")
    elif limit is not None:
        parts.append(f"cap {limit:.0f}W")
    if clk is not None:
        parts.append(f"{clk:.0f}MHz")
    if temp is not None:
        parts.append(f"{temp:.0f}C")
    return "  ".join(parts) if parts else "no telemetry exposed"


def bench_matmul(n: int, dtype, iters: int) -> tuple[float, float, str]:
    """Returns (TFLOPS, peak GiB, telemetry sampled while still hot).

    A matmul of two n x n matrices is 2*n^3 flops - n^3 multiplies and
    n^3 adds. That identity is the whole measurement.
    """
    torch.cuda.reset_peak_memory_stats()
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Warmup: the first calls pay for kernel autotuning and clock ramp.
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    # CUDA is async - without this sync we'd time the enqueue, not the work.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    telem = gpu_telemetry()  # sample immediately, before the card cools
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del a, b
    torch.cuda.empty_cache()
    return (iters * 2 * n**3) / elapsed / 1e12, peak, telem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    device_report()
    print(f"idle        {gpu_telemetry()}\n")

    runs = [
        ("bf16", torch.bfloat16, False),
        ("fp16", torch.float16, False),
        ("tf32", torch.float32, True),
        ("fp32", torch.float32, False),
    ]
    if args.reverse:
        runs.reverse()

    print(f"{'':>6}  {'TFLOPS':>8}  {'peak':>6}   telemetry")
    for name, dtype, tf32 in runs:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        best, peak, telem = 0.0, 0.0, ""
        try:
            for _ in range(args.repeat):
                t, peak, telem = bench_matmul(args.n, dtype, args.iters)
                best = max(best, t)
                if args.repeat > 1:
                    time.sleep(args.cooldown)
            print(f"{name:>6}  {best:8.1f}  {peak:5.1f}G   {telem}")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:>6}  {'OOM':>8}")
        time.sleep(args.cooldown)

    print()
    print("Reading it:")
    print("  bf16 and fp16 should be near-identical - both are tensor-core")
    print("  paths at the same bit width. A large gap means either thermal")
    print("  drift across the run (check with --reverse) or fp16-with-fp32-")
    print("  accumulate being half-rated in silicon.")
    print()
    print("  tf32 vs fp32 is the same fp32 tensors, but matmuls internally")
    print("  truncated to 19 bits of mantissa. Nearly free accuracy-wise for")
    print("  deep learning, and it is why nobody trains in true fp32 anymore.")


if __name__ == "__main__":
    main()
