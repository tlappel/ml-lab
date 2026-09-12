"""Matmul benchmark with memory and power reporting.

Run plugged in, with Windows Power mode on Best Performance. On battery this
card turns in roughly 60% of its real throughput, which is an easy way to draw
wrong conclusions for months.

    python bench.py
"""

import time

import torch


def device_report() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available. If torch is installed, it's probably the wrong "
            "build — Blackwell needs the cu128 wheels:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu128"
        )

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024**3
    # Compute capability 12.x is Blackwell. 8.9 is Ada, 8.6 Ampere.
    cc = f"{props.major}.{props.minor}"

    print(f"device      {props.name}")
    print(f"capability  sm_{props.major}{props.minor}  (compute {cc})")
    print(f"vram        {total_gb:.1f} GiB")
    print(f"sms         {props.multi_processor_count}")
    print(f"torch       {torch.__version__}")
    print(f"cuda build  {torch.version.cuda}")
    print()


def power_draw() -> str:
    """Current draw vs. cap, if the driver will tell us."""
    try:
        watts = torch.cuda.power_draw(0) / 1000.0
        return f"{watts:.0f}W"
    except Exception:
        return "n/a"


def bench_matmul(n: int = 8192, dtype=torch.bfloat16, iters: int = 50) -> float:
    """Time square matmuls and return achieved TFLOPS.

    A matmul of two n x n matrices is 2*n^3 floating point operations —
    n^3 multiplies and n^3 adds. That identity is the whole measurement.
    """
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)

    # Warmup. The first calls pay for kernel autotuning and clock ramp, and
    # including them would understate the card badly.
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        torch.matmul(a, b)
    # CUDA calls are async — without this sync we'd time the enqueue, not the work.
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = iters * 2 * n**3
    return flops / elapsed / 1e12


def main() -> None:
    device_report()

    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)):
        # fp32 at 8192 needs 3 x 256MiB; fine, but keep an eye on it.
        try:
            tflops = bench_matmul(dtype=dtype)
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"{name:>5}  {tflops:7.1f} TFLOPS   peak {mem:4.1f} GiB   {power_draw()}")
        except torch.cuda.OutOfMemoryError:
            print(f"{name:>5}  OOM")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print()
    print("fp32 being ~10x slower than bf16 is correct — tensor cores only")
    print("accelerate the reduced-precision paths. That gap is the reason")
    print("everything in modern ML trains in bf16.")


if __name__ == "__main__":
    main()
