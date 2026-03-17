# SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark: pre-optimisation (PyTorch reference) vs
post-optimisation (SYCL C++ kernels via ``lmcache.xpu_ops``).

The test file exercises the same three hot-path functions that the
SYCL kernels implement and compares:

  * **PyTorch reference** — a pure-PyTorch implementation that
    performs the exact same data movement using ``index_select``,
    ``index_copy_``, and tensor slicing.  This represents the
    "pre-optimisation" baseline (equivalent to what a naïve
    line-by-line CUDA→SYCL port would achieve on Intel XPU, since
    PyTorch's XPU backend already uses SYCL under the hood).

  * **Optimised SYCL kernels** — the hand-tuned SYCL kernels from
    ``lmcache.xpu_ops`` that use compile-time template
    specialisation, larger work-groups, sub-group alignment, and
    loop unrolling for maximum Intel XPU throughput.

Usage
-----
Run on a machine with an Intel XPU device::

    pytest tests/v1/test_sycl_perf_benchmark.py -xvs

If no XPU device is available the tests are automatically skipped.
The benchmark results are printed to stdout in a human-readable
table showing median latency and throughput for each kernel ×
direction × problem size.
"""

# Standard
from typing import List, Tuple
import random

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the entire module when no XPU device is detected.
# ---------------------------------------------------------------------------
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("Intel XPU device not available", allow_module_level=True)

try:
    # First Party
    import lmcache.xpu_ops as xpu_ops  # noqa: E402
except ImportError:
    pytest.skip(
        "lmcache.xpu_ops not built (BUILD_WITH_SYCL=1)",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
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
WARMUP_ITERS = 5
BENCH_ITERS = 20


def _sync():
    """Synchronize the XPU device."""
    torch.xpu.synchronize()


def _make_slot_mapping(num_tokens: int) -> torch.Tensor:
    slots = random.sample(range(0, PAGE_BUFFER_SIZE), num_tokens)
    return torch.tensor(slots, dtype=torch.int64, device=DEVICE)


def _make_paged_kv(
    fmt: "xpu_ops.GPUKVFormat",
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return (kv_cache_list, kv_cache_pointers)."""
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
    """
    Return the **median** execution time in milliseconds.

    Uses XPU events for accurate device-side timing.
    """
    # Warmup
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


# ===================================================================
# Pure-PyTorch reference implementations ("pre-optimisation")
# ===================================================================
def _pytorch_multi_layer_d2h(
    kv_cache: List[torch.Tensor],
    slot_mapping: torch.Tensor,
    key_value: torch.Tensor,
    fmt: "xpu_ops.GPUKVFormat",
):
    """
    PyTorch-only D2H multi-layer transfer (reference / baseline).

    Equivalent to ``multi_layer_kv_transfer(..., D2H)``.
    """
    use_mla = fmt in (
        xpu_ops.GPUKVFormat.NL_X_NB_BS_HS,
        xpu_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    num_tokens = slot_mapping.size(0)
    block_indices = slot_mapping // BLOCK_SIZE
    block_offsets = slot_mapping % BLOCK_SIZE
    flat_indices = block_indices * BLOCK_SIZE + block_offsets

    for layer_id in range(NUM_LAYERS):
        layer_kv = kv_cache[layer_id]
        if use_mla:
            # shape: [num_blocks, block_size, head_size]
            flat = layer_kv.reshape(-1, HEAD_SIZE)
            key_value[0, layer_id, :num_tokens] = flat[flat_indices]
        else:
            if fmt == xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                # [2, num_blocks, block_size, num_heads, head_size]
                k_flat = layer_kv[0].reshape(-1, HIDDEN_DIM)
                v_flat = layer_kv[1].reshape(-1, HIDDEN_DIM)
            else:
                # [num_blocks, 2, block_size, num_heads, head_size]
                k_flat = layer_kv[:, 0].reshape(-1, HIDDEN_DIM)
                v_flat = layer_kv[:, 1].reshape(-1, HIDDEN_DIM)
            key_value[0, layer_id, :num_tokens] = k_flat[flat_indices]
            key_value[1, layer_id, :num_tokens] = v_flat[flat_indices]


def _pytorch_multi_layer_h2d(
    kv_cache: List[torch.Tensor],
    slot_mapping: torch.Tensor,
    key_value: torch.Tensor,
    fmt: "xpu_ops.GPUKVFormat",
):
    """
    PyTorch-only H2D multi-layer transfer (reference / baseline).

    Equivalent to ``multi_layer_kv_transfer(..., H2D)``.
    """
    use_mla = fmt in (
        xpu_ops.GPUKVFormat.NL_X_NB_BS_HS,
        xpu_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
    )
    num_tokens = slot_mapping.size(0)
    block_indices = slot_mapping // BLOCK_SIZE
    block_offsets = slot_mapping % BLOCK_SIZE
    flat_indices = block_indices * BLOCK_SIZE + block_offsets

    for layer_id in range(NUM_LAYERS):
        layer_kv = kv_cache[layer_id]
        if use_mla:
            flat = layer_kv.reshape(-1, HEAD_SIZE)
            flat[flat_indices] = key_value[0, layer_id, :num_tokens]
        else:
            if fmt == xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
                k_flat = layer_kv[0].reshape(-1, HIDDEN_DIM)
                v_flat = layer_kv[1].reshape(-1, HIDDEN_DIM)
            else:
                k_flat = layer_kv[:, 0].reshape(-1, HIDDEN_DIM)
                v_flat = layer_kv[:, 1].reshape(-1, HIDDEN_DIM)
            k_flat[flat_indices] = key_value[0, layer_id, :num_tokens]
            v_flat[flat_indices] = key_value[1, layer_id, :num_tokens]


def _pytorch_single_layer_d2h(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    tmp_buf: torch.Tensor,
    fmt: "xpu_ops.GPUKVFormat",
):
    """
    PyTorch-only D2H single-layer transfer (reference / baseline).

    Equivalent to ``single_layer_kv_transfer(..., D2H)``.
    """
    block_indices = slot_mapping // BLOCK_SIZE
    block_offsets = slot_mapping % BLOCK_SIZE
    flat_indices = block_indices * BLOCK_SIZE + block_offsets

    if fmt == xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        k_flat = kv_layer[0].reshape(-1, HIDDEN_DIM)
        v_flat = kv_layer[1].reshape(-1, HIDDEN_DIM)
    else:
        k_flat = kv_layer[:, 0].reshape(-1, HIDDEN_DIM)
        v_flat = kv_layer[:, 1].reshape(-1, HIDDEN_DIM)

    # token-major: [num_tokens, 2, hidden_dim]
    tmp_buf[:, 0, :] = k_flat[flat_indices]
    tmp_buf[:, 1, :] = v_flat[flat_indices]


def _pytorch_single_layer_h2d(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    tmp_buf: torch.Tensor,
    fmt: "xpu_ops.GPUKVFormat",
):
    """PyTorch-only H2D single-layer transfer (reference / baseline)."""
    block_indices = slot_mapping // BLOCK_SIZE
    block_offsets = slot_mapping % BLOCK_SIZE
    flat_indices = block_indices * BLOCK_SIZE + block_offsets

    if fmt == xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        k_flat = kv_layer[0].reshape(-1, HIDDEN_DIM)
        v_flat = kv_layer[1].reshape(-1, HIDDEN_DIM)
    else:
        k_flat = kv_layer[:, 0].reshape(-1, HIDDEN_DIM)
        v_flat = kv_layer[:, 1].reshape(-1, HIDDEN_DIM)

    k_flat[flat_indices] = tmp_buf[:, 0, :]
    v_flat[flat_indices] = tmp_buf[:, 1, :]


# ===================================================================
# Benchmark tests
# ===================================================================
def _print_result(
    kernel: str,
    direction: str,
    num_tokens: int,
    ref_ms: float,
    opt_ms: float,
):
    speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
    data_bytes = NUM_LAYERS * num_tokens * HIDDEN_DIM * 2 * DTYPE.itemsize
    ref_gbps = data_bytes / (ref_ms * 1e-3) / 1e9
    opt_gbps = data_bytes / (opt_ms * 1e-3) / 1e9
    print(
        f"  {kernel:<35s}  tokens={num_tokens:<5d}  "
        f"ref={ref_ms:8.3f}ms ({ref_gbps:6.1f} GB/s)  "
        f"opt={opt_ms:8.3f}ms ({opt_gbps:6.1f} GB/s)  "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.parametrize("num_tokens", [256, 1024, 4096])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ],
)
def test_bench_multi_layer_d2h(num_tokens, gpu_kv_format):
    """Benchmark multi_layer_kv_transfer D2H: PyTorch ref vs SYCL."""
    kv_cache, ptrs = _make_paged_kv(gpu_kv_format)
    slot_mapping = _make_slot_mapping(num_tokens)

    kv_shape = torch.Size([2, NUM_LAYERS, num_tokens, HIDDEN_DIM])
    key_value_ref = torch.empty(kv_shape, dtype=DTYPE, device=DEVICE)
    key_value_opt = torch.empty(kv_shape, dtype=DTYPE, device=DEVICE)

    # -- Reference (PyTorch) -----------------------------------------
    ref_ms = _bench(
        lambda: _pytorch_multi_layer_d2h(
            kv_cache, slot_mapping, key_value_ref, gpu_kv_format
        )
    )

    # -- Optimised SYCL kernel ---------------------------------------
    opt_ms = _bench(
        lambda: xpu_ops.multi_layer_kv_transfer(
            key_value_opt,
            ptrs,
            slot_mapping,
            kv_cache[0].device,
            PAGE_BUFFER_SIZE,
            xpu_ops.TransferDirection.D2H,
            gpu_kv_format,
            BLOCK_SIZE,
        )
    )

    _print_result("multi_layer D2H", str(gpu_kv_format), num_tokens, ref_ms, opt_ms)


@pytest.mark.parametrize("num_tokens", [256, 1024, 4096])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ],
)
def test_bench_multi_layer_h2d(num_tokens, gpu_kv_format):
    """Benchmark multi_layer_kv_transfer H2D: PyTorch ref vs SYCL."""
    kv_cache, ptrs = _make_paged_kv(gpu_kv_format)
    slot_mapping = _make_slot_mapping(num_tokens)

    kv_shape = torch.Size([2, NUM_LAYERS, num_tokens, HIDDEN_DIM])
    key_value_ref = torch.rand(kv_shape, dtype=DTYPE, device=DEVICE)
    key_value_opt = key_value_ref.clone()

    kv_cache_ref, _ = _make_paged_kv(gpu_kv_format)
    kv_cache_opt, ptrs_opt = _make_paged_kv(gpu_kv_format)

    ref_ms = _bench(
        lambda: _pytorch_multi_layer_h2d(
            kv_cache_ref, slot_mapping, key_value_ref, gpu_kv_format
        )
    )

    opt_ms = _bench(
        lambda: xpu_ops.multi_layer_kv_transfer(
            key_value_opt,
            ptrs_opt,
            slot_mapping,
            kv_cache_opt[0].device,
            PAGE_BUFFER_SIZE,
            xpu_ops.TransferDirection.H2D,
            gpu_kv_format,
            BLOCK_SIZE,
        )
    )

    _print_result("multi_layer H2D", str(gpu_kv_format), num_tokens, ref_ms, opt_ms)


@pytest.mark.parametrize("num_tokens", [256, 1024, 4096])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ],
)
def test_bench_single_layer_d2h(num_tokens, gpu_kv_format):
    """Benchmark single_layer_kv_transfer D2H: PyTorch ref vs SYCL."""
    kv_cache, _ = _make_paged_kv(gpu_kv_format)
    slot_mapping = _make_slot_mapping(num_tokens)
    layer_kv = kv_cache[0]

    tmp_ref = torch.empty((num_tokens, 2, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)
    tmp_opt = torch.empty((num_tokens, 2, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)

    ref_ms = _bench(
        lambda: _pytorch_single_layer_d2h(
            layer_kv, slot_mapping, tmp_ref, gpu_kv_format
        )
    )

    opt_ms = _bench(
        lambda: xpu_ops.single_layer_kv_transfer(
            tmp_opt,
            layer_kv,
            slot_mapping,
            xpu_ops.TransferDirection.D2H,
            gpu_kv_format,
            True,  # token_major
        )
    )

    data_bytes = num_tokens * HIDDEN_DIM * 2 * DTYPE.itemsize
    speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
    ref_gbps = data_bytes / (ref_ms * 1e-3) / 1e9
    opt_gbps = data_bytes / (opt_ms * 1e-3) / 1e9
    print(
        f"  {'single_layer D2H':<35s}  tokens={num_tokens:<5d}  "
        f"ref={ref_ms:8.3f}ms ({ref_gbps:6.1f} GB/s)  "
        f"opt={opt_ms:8.3f}ms ({opt_gbps:6.1f} GB/s)  "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.parametrize("num_tokens", [256, 1024, 4096])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        xpu_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        xpu_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ],
)
def test_bench_single_layer_h2d(num_tokens, gpu_kv_format):
    """Benchmark single_layer_kv_transfer H2D: PyTorch ref vs SYCL."""
    kv_cache, _ = _make_paged_kv(gpu_kv_format)
    slot_mapping = _make_slot_mapping(num_tokens)
    layer_kv = kv_cache[0]
    layer_kv_opt, _ = _make_paged_kv(gpu_kv_format)
    layer_kv_opt = layer_kv_opt[0]

    tmp_buf = torch.rand((num_tokens, 2, HIDDEN_DIM), dtype=DTYPE, device=DEVICE)

    ref_ms = _bench(
        lambda: _pytorch_single_layer_h2d(
            layer_kv, slot_mapping, tmp_buf, gpu_kv_format
        )
    )

    opt_ms = _bench(
        lambda: xpu_ops.single_layer_kv_transfer(
            tmp_buf,
            layer_kv_opt,
            slot_mapping,
            xpu_ops.TransferDirection.H2D,
            gpu_kv_format,
            True,  # token_major
        )
    )

    data_bytes = num_tokens * HIDDEN_DIM * 2 * DTYPE.itemsize
    speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
    ref_gbps = data_bytes / (ref_ms * 1e-3) / 1e9
    opt_gbps = data_bytes / (opt_ms * 1e-3) / 1e9
    print(
        f"  {'single_layer H2D':<35s}  tokens={num_tokens:<5d}  "
        f"ref={ref_ms:8.3f}ms ({ref_gbps:6.1f} GB/s)  "
        f"opt={opt_ms:8.3f}ms ({opt_gbps:6.1f} GB/s)  "
        f"speedup={speedup:.2f}x"
    )


@pytest.mark.parametrize("num_tokens", [256, 1024, 4096])
def test_bench_multi_layer_mla_d2h(num_tokens):
    """Benchmark multi_layer_kv_transfer MLA D2H: PyTorch ref vs SYCL."""
    fmt = xpu_ops.GPUKVFormat.NL_X_NB_BS_HS
    kv_cache, ptrs = _make_paged_kv(fmt)
    slot_mapping = _make_slot_mapping(num_tokens)

    kv_shape = torch.Size([1, NUM_LAYERS, num_tokens, HEAD_SIZE])
    key_value_ref = torch.empty(kv_shape, dtype=DTYPE, device=DEVICE)
    key_value_opt = torch.empty(kv_shape, dtype=DTYPE, device=DEVICE)

    ref_ms = _bench(
        lambda: _pytorch_multi_layer_d2h(kv_cache, slot_mapping, key_value_ref, fmt)
    )

    opt_ms = _bench(
        lambda: xpu_ops.multi_layer_kv_transfer(
            key_value_opt,
            ptrs,
            slot_mapping,
            kv_cache[0].device,
            PAGE_BUFFER_SIZE,
            xpu_ops.TransferDirection.D2H,
            fmt,
            BLOCK_SIZE,
        )
    )

    data_bytes = NUM_LAYERS * num_tokens * HEAD_SIZE * DTYPE.itemsize
    speedup = ref_ms / opt_ms if opt_ms > 0 else float("inf")
    ref_gbps = data_bytes / (ref_ms * 1e-3) / 1e9
    opt_gbps = data_bytes / (opt_ms * 1e-3) / 1e9
    print(
        f"  {'multi_layer MLA D2H':<35s}  tokens={num_tokens:<5d}  "
        f"ref={ref_ms:8.3f}ms ({ref_gbps:6.1f} GB/s)  "
        f"opt={opt_ms:8.3f}ms ({opt_gbps:6.1f} GB/s)  "
        f"speedup={speedup:.2f}x"
    )
