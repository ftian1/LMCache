# SPDX-License-Identifier: Apache-2.0
"""Performance comparison tests between VLLMPagedMemXPUConnectorV2
(pure PyTorch index_select/index_copy_ operations) and
VLLMPagedMemLayerwiseGPUConnector (CUDA kernels with CUDA streams).

Both connectors transfer KV cache data between paged GPU memory and
host/device buffers, but use different strategies:

- XPU connector: processes ALL layers at once using PyTorch tensor ops
- Layerwise GPU connector: processes one layer at a time using custom
  CUDA kernels and async CUDA streams

Run with:
    pytest -xvs tests/v1/test_gpu_connector_perf_comparison.py
    pytest --benchmark-only tests/v1/test_gpu_connector_perf_comparison.py
"""

# Standard
from contextlib import nullcontext
from unittest.mock import patch
import random
import threading
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)
from lmcache.v1.gpu_connector.xpu_connectors import VLLMPagedMemXPUConnectorV2
from lmcache.v1.memory_management import (
    MemoryFormat,
    PagedTensorMemoryAllocator,
    PinMemoryAllocator,
    TensorMemoryAllocator,
)

if torch.cuda.is_available():
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        lmc_ops = None
else:
    lmc_ops = None

# Mock c_ops when not available
if lmc_ops is None:

    class MockGPUKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2

    class MockCOps:
        GPUKVFormat = MockGPUKVFormat

    lmc_ops = MockCOps()

# Local
from .utils import (
    generate_kv_cache_paged_list_tensors,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def patch_pin_allocator():
    """Patch PinMemoryAllocator to avoid cudaHostRegister issues in tests."""

    def fake_pin_init(self, size: int, use_paging: bool = False, **kwargs):
        self._unregistered = False
        self.buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)

        if use_paging:
            assert "shapes" in kwargs, (
                "shapes must be specified for paged memory allocator"
            )
            assert "dtypes" in kwargs, (
                "dtypes must be specified for paged memory allocator"
            )
            assert "fmt" in kwargs, "fmt must be specified for paged memory allocator"
            self.allocator = PagedTensorMemoryAllocator(
                tensor=self.buffer,
                shapes=kwargs["shapes"],
                dtypes=kwargs["dtypes"],
                fmt=kwargs["fmt"],
            )
        else:
            self.allocator = TensorMemoryAllocator(self.buffer)

        self.host_mem_lock = threading.Lock() if not use_paging else nullcontext()

    def fake_pin_close(self):
        if not self._unregistered:
            torch.cuda.synchronize()
            self._unregistered = True

    with (
        patch(
            "lmcache.v1.memory_management.PinMemoryAllocator.__init__",
            fake_pin_init,
        ),
        patch(
            "lmcache.v1.memory_management.PinMemoryAllocator.close",
            fake_pin_close,
        ),
    ):
        yield


@pytest.fixture()
def mock_xpu_sync():
    """Mock ``torch.xpu.synchronize`` with ``torch.cuda.synchronize``
    so the XPU connector can run on CUDA hardware for benchmarking."""
    xpu_ns = types.SimpleNamespace(synchronize=torch.cuda.synchronize)
    with patch.object(torch, "xpu", xpu_ns, create=True):
        yield


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
NUM_BLOCKS = 100
BLOCK_SIZE = 16
NUM_LAYERS = 32
NUM_HEADS = 8
HEAD_SIZE = 128
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE
DEVICE = "cuda"
CHUNK_SIZE = 256
NUM_TOKENS = 256  # single chunk for cleaner benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slot_mapping(num_blocks: int, block_size: int, num_tokens: int):
    """Create a random slot mapping tensor on CUDA."""
    slots = random.sample(range(0, num_blocks * block_size), num_tokens)
    return torch.tensor(slots, device=DEVICE, dtype=torch.int64)


def _run_xpu_from_gpu(connector, memory_obj, slot_mapping, kvcaches, start, end):
    """Execute XPU connector from_gpu and synchronize."""
    connector.from_gpu(
        memory_obj,
        start,
        end,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        offset=0,
    )
    torch.cuda.synchronize()


def _run_xpu_to_gpu(connector, memory_obj, slot_mapping, kvcaches, start, end):
    """Execute XPU connector to_gpu and synchronize."""
    connector.to_gpu(
        memory_obj,
        start,
        end,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        offset=0,
    )
    torch.cuda.synchronize()


def _run_layerwise_from_gpu(
    connector, memory_objs, starts, ends, kvcaches, slot_mapping
):
    """Execute layerwise GPU connector batched_from_gpu (full pipeline)."""
    gen = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        sync=True,
    )
    for _ in range(NUM_LAYERS + 1):
        next(gen)
    torch.cuda.synchronize()


def _run_layerwise_to_gpu(connector, memory_objs, starts, ends, kvcaches, slot_mapping):
    """Execute layerwise GPU connector batched_to_gpu (full pipeline)."""
    consumer = connector.batched_to_gpu(
        starts,
        ends,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        sync=True,
    )
    next(consumer)
    for layer_id in range(NUM_LAYERS):
        consumer.send(memory_objs[layer_id])
    next(consumer)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Benchmark: XPU connector  (pure PyTorch index_select / index_copy_)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_xpu_connector_from_gpu_bench(benchmark, mock_xpu_sync):
    """Benchmark VLLMPagedMemXPUConnectorV2.from_gpu (GPU → memory obj).

    Uses pure PyTorch index_select to gather KV data from all layers
    at once.
    """
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    kvcaches = generate_kv_cache_paged_list_tensors(NUM_BLOCKS, DEVICE, BLOCK_SIZE)
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    connector = VLLMPagedMemXPUConnectorV2(HIDDEN_DIM, NUM_LAYERS)
    shape = connector.get_shape(CHUNK_SIZE)
    dtype = kvcaches[0][0].dtype

    memory_obj = allocator.allocate(shape, dtype)

    benchmark.pedantic(
        _run_xpu_from_gpu,
        args=(connector, memory_obj, slot_mapping, kvcaches, 0, CHUNK_SIZE),
        rounds=50,
        iterations=50,
        warmup_rounds=5,
    )

    allocator.free(memory_obj)
    assert allocator.memcheck()
    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_xpu_connector_to_gpu_bench(benchmark, mock_xpu_sync):
    """Benchmark VLLMPagedMemXPUConnectorV2.to_gpu (memory obj → GPU).

    Uses pure PyTorch index_copy_ to scatter KV data to all layers
    at once.
    """
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    kvcaches_src = generate_kv_cache_paged_list_tensors(NUM_BLOCKS, DEVICE, BLOCK_SIZE)
    kvcaches_dst = generate_kv_cache_paged_list_tensors(NUM_BLOCKS, DEVICE, BLOCK_SIZE)
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    connector = VLLMPagedMemXPUConnectorV2(HIDDEN_DIM, NUM_LAYERS)
    shape = connector.get_shape(CHUNK_SIZE)
    dtype = kvcaches_src[0][0].dtype

    # First populate memory_obj with data from src
    memory_obj = allocator.allocate(shape, dtype)
    connector.from_gpu(
        memory_obj,
        0,
        CHUNK_SIZE,
        kvcaches=kvcaches_src,
        slot_mapping=slot_mapping,
        offset=0,
    )
    torch.cuda.synchronize()

    benchmark.pedantic(
        _run_xpu_to_gpu,
        args=(connector, memory_obj, slot_mapping, kvcaches_dst, 0, CHUNK_SIZE),
        rounds=50,
        iterations=50,
        warmup_rounds=5,
    )

    allocator.free(memory_obj)
    assert allocator.memcheck()
    allocator.close()


# ---------------------------------------------------------------------------
# Benchmark: Layerwise GPU connector  (CUDA kernels + CUDA streams)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_layerwise_gpu_connector_from_gpu_bench(benchmark):
    """Benchmark VLLMPagedMemLayerwiseGPUConnector.batched_from_gpu
    (GPU → memory objs, layer by layer).

    Uses custom CUDA kernels (single_layer_kv_transfer) with CUDA
    streams for async layer-by-layer transfer.
    """
    gpu_kv_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    kvcaches = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = kvcaches[0][0].dtype
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    connector = VLLMPagedMemLayerwiseGPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=DEVICE,
    )

    starts = [0]
    ends = [CHUNK_SIZE]

    # Pre-allocate memory objects in [layers][chunks] format
    shape_single_layer = connector.get_shape(CHUNK_SIZE)
    memory_objs = [
        [allocator.allocate(shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D)]
        for _ in range(NUM_LAYERS)
    ]

    def _bench_fn():
        _run_layerwise_from_gpu(
            connector, memory_objs, starts, ends, kvcaches, slot_mapping
        )

    benchmark.pedantic(
        _bench_fn,
        rounds=50,
        iterations=50,
        warmup_rounds=5,
    )

    for layer_objs in memory_objs:
        for obj in layer_objs:
            obj.ref_count_down()
    assert allocator.memcheck()
    assert connector.gpu_buffer_allocator.memcheck()
    allocator.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_layerwise_gpu_connector_to_gpu_bench(benchmark):
    """Benchmark VLLMPagedMemLayerwiseGPUConnector.batched_to_gpu
    (memory objs → GPU, layer by layer).

    Uses custom CUDA kernels (single_layer_kv_transfer) with CUDA
    streams for async layer-by-layer transfer.
    """
    gpu_kv_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    allocator = PinMemoryAllocator(1024 * 1024 * 1024)
    kvcaches_src = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    kvcaches_dst = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = kvcaches_src[0][0].dtype
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    connector = VLLMPagedMemLayerwiseGPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=DEVICE,
    )

    starts = [0]
    ends = [CHUNK_SIZE]

    # Pre-allocate memory objects in [layers][chunks] format and populate via from_gpu
    shape_single_layer = connector.get_shape(CHUNK_SIZE)
    memory_objs = [
        [allocator.allocate(shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D)]
        for _ in range(NUM_LAYERS)
    ]

    _run_layerwise_from_gpu(
        connector, memory_objs, starts, ends, kvcaches_src, slot_mapping
    )

    def _bench_fn():
        _run_layerwise_to_gpu(
            connector, memory_objs, starts, ends, kvcaches_dst, slot_mapping
        )

    benchmark.pedantic(
        _bench_fn,
        rounds=50,
        iterations=50,
        warmup_rounds=5,
    )

    for layer_objs_list in memory_objs:
        for obj in layer_objs_list:
            obj.ref_count_down()
    assert allocator.memcheck()
    assert connector.gpu_buffer_allocator.memcheck()
    allocator.close()


# ---------------------------------------------------------------------------
# Direct comparison: side-by-side timing with torch.cuda.Event
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_xpu_vs_layerwise_gpu_connector_comparison(mock_xpu_sync):
    """Side-by-side timing comparison of XPU connector (pure PyTorch)
    vs Layerwise GPU connector (CUDA kernels).

    Prints a summary table with from_gpu and to_gpu timings for both
    approaches.  This test always passes — it is informational only.
    """
    gpu_kv_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    num_warmup = 5
    num_runs = 50

    pin_allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    kvcaches_src = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    kvcaches_dst_xpu = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    kvcaches_dst_layerwise = generate_kv_cache_paged_list_tensors(
        num_blocks=NUM_BLOCKS,
        device=DEVICE,
        block_size=BLOCK_SIZE,
        gpu_kv_format=gpu_kv_format,
    )
    dtype = kvcaches_src[0][0].dtype
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    # ---- XPU connector setup ----
    xpu_connector = VLLMPagedMemXPUConnectorV2(HIDDEN_DIM, NUM_LAYERS)
    xpu_shape = xpu_connector.get_shape(CHUNK_SIZE)
    xpu_mem_obj = pin_allocator.allocate(xpu_shape, dtype)

    # ---- Layerwise GPU connector setup ----
    lw_connector = VLLMPagedMemLayerwiseGPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=DEVICE,
    )
    lw_shape = lw_connector.get_shape(CHUNK_SIZE)
    lw_mem_objs = [
        [pin_allocator.allocate(lw_shape, dtype, fmt=MemoryFormat.KV_T2D)]
        for _ in range(NUM_LAYERS)
    ]

    starts = [0]
    ends = [CHUNK_SIZE]

    def _timed_cuda(fn, warmup: int, runs: int) -> float:
        """Return median GPU time in milliseconds."""
        # Warmup
        for _ in range(warmup):
            fn()

        times = []
        for _ in range(runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

        times.sort()
        return times[len(times) // 2]  # median

    # ---- from_gpu benchmarks ----
    xpu_from_gpu_ms = _timed_cuda(
        lambda: _run_xpu_from_gpu(
            xpu_connector, xpu_mem_obj, slot_mapping, kvcaches_src, 0, CHUNK_SIZE
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    lw_from_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_from_gpu(
            lw_connector, lw_mem_objs, starts, ends, kvcaches_src, slot_mapping
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    # ---- to_gpu benchmarks ----
    # Populate XPU memory obj first
    _run_xpu_from_gpu(
        xpu_connector, xpu_mem_obj, slot_mapping, kvcaches_src, 0, CHUNK_SIZE
    )
    # Populate layerwise memory objs first
    _run_layerwise_from_gpu(
        lw_connector, lw_mem_objs, starts, ends, kvcaches_src, slot_mapping
    )

    xpu_to_gpu_ms = _timed_cuda(
        lambda: _run_xpu_to_gpu(
            xpu_connector, xpu_mem_obj, slot_mapping, kvcaches_dst_xpu, 0, CHUNK_SIZE
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    lw_to_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_to_gpu(
            lw_connector,
            lw_mem_objs,
            starts,
            ends,
            kvcaches_dst_layerwise,
            slot_mapping,
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    # ---- Print comparison table ----
    header = (
        f"\n{'=' * 70}\n"
        f"  Performance Comparison: XPU Connector vs Layerwise GPU Connector\n"
        f"  Config: {NUM_LAYERS} layers, {NUM_HEADS} heads, "
        f"head_size={HEAD_SIZE}, chunk={CHUNK_SIZE} tokens\n"
        f"{'=' * 70}"
    )
    row_fmt = "  {:<35s} {:>10.3f} ms  {:>10.3f} ms  {:>8.2f}x"
    print(header)
    print(
        f"  {'Operation':<35s} {'XPU (PyTorch)':>13s}  "
        f"{'Layerwise (CUDA)':>13s}  {'Speedup':>8s}"
    )
    print(f"  {'-' * 35} {'-' * 13}  {'-' * 13}  {'-' * 8}")
    print(
        row_fmt.format(
            "from_gpu (GPU → host)",
            xpu_from_gpu_ms,
            lw_from_gpu_ms,
            xpu_from_gpu_ms / lw_from_gpu_ms if lw_from_gpu_ms > 0 else float("inf"),
        )
    )
    print(
        row_fmt.format(
            "to_gpu (host → GPU)",
            xpu_to_gpu_ms,
            lw_to_gpu_ms,
            xpu_to_gpu_ms / lw_to_gpu_ms if lw_to_gpu_ms > 0 else float("inf"),
        )
    )
    print(f"{'=' * 70}\n")

    # Cleanup
    pin_allocator.free(xpu_mem_obj)
    for layer_objs_list in lw_mem_objs:
        for obj in layer_objs_list:
            obj.ref_count_down()

    assert pin_allocator.memcheck()
    assert lw_connector.gpu_buffer_allocator.memcheck()
    pin_allocator.close()
