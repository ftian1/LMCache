# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark for comparing two compiled versions of the
SYCL memory kernels in ``lmcache.xpu_ops``.

Designed to measure the performance difference between:

  * **Version A** (commit d716c3d) -- initial SYCL port: WG=128,
    runtime ``if (DIRECTION)`` branching, no sub-group rounding.

  * **Version B** (commit 4d02a06) -- optimized SYCL: WG=256,
    ``if constexpr (DIRECTION)`` compile-time branching, sub-group
    alignment via ``round_up_to_sg()``, ``-ffast-math
    -funroll-loops``.

Workflow
--------
1.  Build Version A (the initial port)::

        git checkout d716c3d -- csrc/sycl/ setup.py
        CXX=icpx BUILD_WITH_SYCL=1 pip install -e . --no-build-isolation

2.  Run the benchmark and save results::

        python tests/v1/test_sycl_perf_benchmark.py --save version_a.json

3.  Build Version B (the optimized version)::

        git checkout 4d02a06 -- csrc/sycl/ setup.py
        CXX=icpx BUILD_WITH_SYCL=1 pip install -e . --no-build-isolation

4.  Run the benchmark and save results::

        python tests/v1/test_sycl_perf_benchmark.py --save version_b.json

5.  Compare the two runs::

        python tests/v1/test_sycl_perf_benchmark.py \\
            --compare version_a.json version_b.json

    This prints a side-by-side table with latencies and speedup.

The file can also be run via ``pytest`` (each kernel/size combination
is a separate test).  When no XPU device is available every test is
automatically skipped.
"""

# Standard
from typing import Dict, List, Tuple
import argparse
import json
import random
import sys

# Third Party
import torch

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def _get_xpu_ops():
    """Try to import lmcache.xpu_ops; return None on failure."""
    try:
        # First Party
        import lmcache.xpu_ops as _ops

        return _ops
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = "xpu"
DTYPE = torch.bfloat16
NUM_HEADS = 8
HEAD_SIZE = 128
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE
BLOCK_SIZE = 16
NUM_LAYERS = 32
NUM_BLOCKS = 1000
PAGE_BUFFER_SIZE = NUM_BLOCKS * BLOCK_SIZE
WARMUP_ITERS = 10
BENCH_ITERS = 50
TOKEN_SIZES = [256, 1024, 4096]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sync():
    """Synchronize the XPU device."""
    torch.xpu.synchronize()


def _make_slot_mapping(num_tokens: int) -> torch.Tensor:
    """Create a random slot mapping on the XPU device.

    Args:
        num_tokens: Number of tokens to create slots for.

    Returns:
        A 1-D int64 tensor of unique slot indices on the XPU device.
    """
    slots = random.sample(range(0, PAGE_BUFFER_SIZE), num_tokens)
    return torch.tensor(slots, dtype=torch.int64, device=DEVICE)


def _make_paged_kv(xpu_ops, fmt) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Create random paged KV caches and a pointer tensor.

    Args:
        xpu_ops: The xpu_ops module (for GPUKVFormat enum values).
        fmt: The GPUKVFormat to use.

    Returns:
        A tuple of (kv_cache_list, kv_cache_pointers).
    """
    use_mla = fmt in (
        xpu_ops.GPUKVFormat.NL_X_NB_BS_HS,
        xpu_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    if use_mla:
        shape = [NUM_BLOCKS, BLOCK_SIZE, HEAD_SIZE]
    elif fmt == xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        shape = [2, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE]
    else:  # NL_X_NB_TWO_BS_NH_HS
        shape = [NUM_BLOCKS, 2, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE]

    kv = [torch.rand(shape, dtype=DTYPE, device=DEVICE) for _ in range(NUM_LAYERS)]
    ptrs = torch.empty(NUM_LAYERS, dtype=torch.int64, device="cpu")
    for i, t in enumerate(kv):
        ptrs[i] = t.data_ptr()
    return kv, ptrs


def _bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> float:
    """Return the **median** execution time in milliseconds.

    Runs *warmup* iterations (discarded), then *iters* timed
    iterations using XPU events.  Returns the median of the timed
    iterations.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        Median execution time in milliseconds.
    """
    for _ in range(warmup):
        fn()
        _sync()

    times: List[float] = []
    for _ in range(iters):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        _sync()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------
def _run_all_benchmarks(xpu_ops) -> Dict[str, float]:
    """Run every kernel x direction x token-size combination.

    Args:
        xpu_ops: The ``lmcache.xpu_ops`` module.

    Returns:
        A dict mapping ``case_name`` to median latency in
        milliseconds.
    """
    results: Dict[str, float] = {}

    formats = [
        ("flash_attn", xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS, False),
        ("flash_infer", xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS, False),
        ("mla", xpu_ops.GPUKVFormat.NL_X_NB_BS_HS, True),
    ]

    for fmt_name, fmt, is_mla in formats:
        for num_tokens in TOKEN_SIZES:
            kv_cache, ptrs = _make_paged_kv(xpu_ops, fmt)
            slot_mapping = _make_slot_mapping(num_tokens)
            kv_dim = HEAD_SIZE if is_mla else HIDDEN_DIM
            kv_or_v = 1 if is_mla else 2
            kv_shape = torch.Size([kv_or_v, NUM_LAYERS, num_tokens, kv_dim])

            # -- multi_layer D2H ----------------------------------
            key_value = torch.empty(kv_shape, dtype=DTYPE, device=DEVICE)
            dev = kv_cache[0].device
            case = f"multi_layer_d2h/{fmt_name}/tok{num_tokens}"
            ms = _bench(
                lambda kv=key_value, p=ptrs, s=slot_mapping, f=fmt, d=dev: (
                    xpu_ops.multi_layer_kv_transfer(
                        kv,
                        p,
                        s,
                        d,
                        PAGE_BUFFER_SIZE,
                        xpu_ops.TransferDirection.D2H,
                        f,
                        BLOCK_SIZE,
                    )
                )
            )
            results[case] = ms
            print(f"  {case:<55s}  {ms:8.3f} ms")

            # -- multi_layer H2D ----------------------------------
            kv_cache2, ptrs2 = _make_paged_kv(xpu_ops, fmt)
            key_value_h2d = torch.rand(kv_shape, dtype=DTYPE, device=DEVICE)
            dev2 = kv_cache2[0].device
            case = f"multi_layer_h2d/{fmt_name}/tok{num_tokens}"
            ms = _bench(
                lambda kv=key_value_h2d, p=ptrs2, s=slot_mapping, f=fmt, d=dev2: (
                    xpu_ops.multi_layer_kv_transfer(
                        kv,
                        p,
                        s,
                        d,
                        PAGE_BUFFER_SIZE,
                        xpu_ops.TransferDirection.H2D,
                        f,
                        BLOCK_SIZE,
                    )
                )
            )
            results[case] = ms
            print(f"  {case:<55s}  {ms:8.3f} ms")

            # -- single_layer D2H / H2D (skip MLA) ---------------
            if not is_mla:
                layer_kv = kv_cache[0]
                tmp = torch.empty(
                    (num_tokens, 2, HIDDEN_DIM),
                    dtype=DTYPE,
                    device=DEVICE,
                )
                case = f"single_layer_d2h/{fmt_name}/tok{num_tokens}"
                ms = _bench(
                    lambda t=tmp, lk=layer_kv, s=slot_mapping, f=fmt: (
                        xpu_ops.single_layer_kv_transfer(
                            t,
                            lk,
                            s,
                            xpu_ops.TransferDirection.D2H,
                            f,
                            True,
                        )
                    )
                )
                results[case] = ms
                print(f"  {case:<55s}  {ms:8.3f} ms")

                layer_kv3, _ = _make_paged_kv(xpu_ops, fmt)
                lk3 = layer_kv3[0]
                case = f"single_layer_h2d/{fmt_name}/tok{num_tokens}"
                ms = _bench(
                    lambda t=tmp, lk=lk3, s=slot_mapping, f=fmt: (
                        xpu_ops.single_layer_kv_transfer(
                            t,
                            lk,
                            s,
                            xpu_ops.TransferDirection.H2D,
                            f,
                            True,
                        )
                    )
                )
                results[case] = ms
                print(f"  {case:<55s}  {ms:8.3f} ms")

    return results


# ---------------------------------------------------------------------------
# Comparison printer
# ---------------------------------------------------------------------------
def _print_comparison(
    results_a: Dict[str, float],
    results_b: Dict[str, float],
    label_a: str,
    label_b: str,
) -> None:
    """Print a side-by-side comparison table.

    Args:
        results_a: Baseline results (version A / old).
        results_b: Candidate results (version B / new).
        label_a: Display label for version A.
        label_b: Display label for version B.
    """
    all_keys = sorted(set(results_a.keys()) | set(results_b.keys()))
    hdr = f"  {'Kernel':<55s}  {label_a:>10s}  {label_b:>10s}  {'Speedup':>8s}"
    print()
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for key in all_keys:
        ms_a = results_a.get(key)
        ms_b = results_b.get(key)
        if ms_a is not None and ms_b is not None and ms_b > 0:
            speedup = ms_a / ms_b
            print(f"  {key:<55s}  {ms_a:9.3f}ms  {ms_b:9.3f}ms  {speedup:7.2f}x")
        elif ms_a is not None:
            print(f"  {key:<55s}  {ms_a:9.3f}ms  {'N/A':>10s}  {'N/A':>8s}")
        elif ms_b is not None:
            print(f"  {key:<55s}  {'N/A':>10s}  {ms_b:9.3f}ms  {'N/A':>8s}")
    print("=" * len(hdr))
    print()

    # Summary
    common = [
        k for k in all_keys if k in results_a and k in results_b and results_b[k] > 0
    ]
    if common:
        speedups = [results_a[k] / results_b[k] for k in common]
        geo_mean = 1.0
        for s in speedups:
            geo_mean *= s
        geo_mean = geo_mean ** (1.0 / len(speedups))
        print(
            f"  Geometric-mean speedup ({label_b} over {label_a}): "
            f"{geo_mean:.2f}x  (across {len(common)} cases)"
        )
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> None:
    """Parse arguments and run benchmarks or comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lmcache.xpu_ops SYCL kernels.  "
            "Run with --save to persist results, then "
            "--compare to diff two saved runs."
        )
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Run benchmarks and save results to JSON file.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help=(
            "Compare two previously saved result files.  "
            "FILE_A is the baseline (version A / old), "
            "FILE_B is the candidate (version B / new)."
        ),
    )
    parser.add_argument(
        "--label-a",
        default="VersionA(old)",
        help="Label for FILE_A in comparison output.",
    )
    parser.add_argument(
        "--label-b",
        default="VersionB(opt)",
        help="Label for FILE_B in comparison output.",
    )
    args = parser.parse_args()

    if args.compare:
        file_a, file_b = args.compare
        with open(file_a) as f:
            ra = json.load(f)
        with open(file_b) as f:
            rb = json.load(f)
        _print_comparison(ra, rb, args.label_a, args.label_b)
        return

    # --- Run benchmarks -------------------------------------------------
    if not _HAS_XPU:
        print("ERROR: No Intel XPU device detected.", file=sys.stderr)
        sys.exit(1)

    xpu_ops = _get_xpu_ops()
    if xpu_ops is None:
        print(
            "ERROR: lmcache.xpu_ops not importable.  Build with BUILD_WITH_SYCL=1.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(
        f"Config: {NUM_LAYERS} layers, {NUM_HEADS} heads, "
        f"head_size={HEAD_SIZE}, block_size={BLOCK_SIZE}, "
        f"num_blocks={NUM_BLOCKS}"
    )
    print(f"Timing: {WARMUP_ITERS} warmup + {BENCH_ITERS} iters (median)\n")

    results = _run_all_benchmarks(xpu_ops)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"\nResults saved to {args.save}")
    else:
        print("\nTip: re-run with --save <file.json> to persist results.")


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------
# When run via pytest, each kernel/size is a separate test that
# simply asserts the kernel completes without error and prints the
# median latency.  Tests are skipped when no XPU is available.

try:
    # Third Party
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

if pytest is not None:
    if not _HAS_XPU:
        pytest.skip(
            "Intel XPU device not available",
            allow_module_level=True,
        )

    _xpu_ops_mod = _get_xpu_ops()
    if _xpu_ops_mod is None:
        pytest.skip(
            "lmcache.xpu_ops not built (BUILD_WITH_SYCL=1)",
            allow_module_level=True,
        )

    # Non-MLA formats
    _NON_MLA_FMTS = [
        _xpu_ops_mod.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        _xpu_ops_mod.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ]

    @pytest.mark.parametrize("num_tokens", TOKEN_SIZES)
    @pytest.mark.parametrize("gpu_kv_format", _NON_MLA_FMTS)
    def test_bench_multi_layer_d2h(num_tokens, gpu_kv_format):
        """Benchmark multi_layer_kv_transfer D2H."""
        kv_cache, ptrs = _make_paged_kv(_xpu_ops_mod, gpu_kv_format)
        slot = _make_slot_mapping(num_tokens)
        kv = torch.empty(
            [2, NUM_LAYERS, num_tokens, HIDDEN_DIM],
            dtype=DTYPE,
            device=DEVICE,
        )
        ms = _bench(
            lambda: _xpu_ops_mod.multi_layer_kv_transfer(
                kv,
                ptrs,
                slot,
                kv_cache[0].device,
                PAGE_BUFFER_SIZE,
                _xpu_ops_mod.TransferDirection.D2H,
                gpu_kv_format,
                BLOCK_SIZE,
            )
        )
        print(f"  multi_layer_d2h  tokens={num_tokens}  {ms:.3f}ms")

    @pytest.mark.parametrize("num_tokens", TOKEN_SIZES)
    @pytest.mark.parametrize("gpu_kv_format", _NON_MLA_FMTS)
    def test_bench_multi_layer_h2d(num_tokens, gpu_kv_format):
        """Benchmark multi_layer_kv_transfer H2D."""
        kv_cache, ptrs = _make_paged_kv(_xpu_ops_mod, gpu_kv_format)
        slot = _make_slot_mapping(num_tokens)
        kv = torch.rand(
            [2, NUM_LAYERS, num_tokens, HIDDEN_DIM],
            dtype=DTYPE,
            device=DEVICE,
        )
        ms = _bench(
            lambda: _xpu_ops_mod.multi_layer_kv_transfer(
                kv,
                ptrs,
                slot,
                kv_cache[0].device,
                PAGE_BUFFER_SIZE,
                _xpu_ops_mod.TransferDirection.H2D,
                gpu_kv_format,
                BLOCK_SIZE,
            )
        )
        print(f"  multi_layer_h2d  tokens={num_tokens}  {ms:.3f}ms")

    @pytest.mark.parametrize("num_tokens", TOKEN_SIZES)
    @pytest.mark.parametrize("gpu_kv_format", _NON_MLA_FMTS)
    def test_bench_single_layer_d2h(num_tokens, gpu_kv_format):
        """Benchmark single_layer_kv_transfer D2H."""
        kv_cache, _ = _make_paged_kv(_xpu_ops_mod, gpu_kv_format)
        slot = _make_slot_mapping(num_tokens)
        tmp = torch.empty((num_tokens, 2, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
        ms = _bench(
            lambda: _xpu_ops_mod.single_layer_kv_transfer(
                tmp,
                kv_cache[0],
                slot,
                _xpu_ops_mod.TransferDirection.D2H,
                gpu_kv_format,
                True,
            )
        )
        print(f"  single_layer_d2h  tokens={num_tokens}  {ms:.3f}ms")

    @pytest.mark.parametrize("num_tokens", TOKEN_SIZES)
    def test_bench_multi_layer_mla_d2h(num_tokens):
        """Benchmark multi_layer_kv_transfer MLA D2H."""
        fmt = _xpu_ops_mod.GPUKVFormat.NL_X_NB_BS_HS
        kv_cache, ptrs = _make_paged_kv(_xpu_ops_mod, fmt)
        slot = _make_slot_mapping(num_tokens)
        kv = torch.empty(
            [1, NUM_LAYERS, num_tokens, HEAD_SIZE],
            dtype=DTYPE,
            device=DEVICE,
        )
        ms = _bench(
            lambda: _xpu_ops_mod.multi_layer_kv_transfer(
                kv,
                ptrs,
                slot,
                kv_cache[0].device,
                PAGE_BUFFER_SIZE,
                _xpu_ops_mod.TransferDirection.D2H,
                fmt,
                BLOCK_SIZE,
            )
        )
        print(f"  multi_layer_mla_d2h  tokens={num_tokens}  {ms:.3f}ms")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _cli()
