# SPDX-License-Identifier: Apache-2.0
"""Performance comparison tests between VLLMPagedMemLayerwiseGPUConnector
(CUDA kernels with CUDA streams) and VLLMPagedMemLayerwiseXPUConnector
(pure PyTorch index_select/index_copy_ with XPU streams).

Both connectors follow the same layerwise generator protocol:

- ``batched_from_gpu(...)`` yields ``num_layers + 1`` times
- ``batched_to_gpu(...)`` yields ``num_layers + 2`` times

The CUDA connector uses custom CUDA kernels
(``lmc_ops.single_layer_kv_transfer``) for data transfer, while the XPU
connector uses pure PyTorch tensor operations (``index_select`` /
``index_copy_``).

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
from lmcache.v1.gpu_connector.xpu_connectors import (
    VLLMPagedMemLayerwiseXPUConnector,
)
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
def mock_xpu_as_cuda():
    """Mock all ``torch.xpu.*`` APIs with their ``torch.cuda.*`` equivalents
    so that ``VLLMPagedMemLayerwiseXPUConnector`` can run on CUDA hardware
    for benchmarking."""
    xpu_ns = types.SimpleNamespace(
        synchronize=torch.cuda.synchronize,
        current_stream=torch.cuda.current_stream,
        Stream=torch.cuda.Stream,
        stream=torch.cuda.stream,
    )
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


def _run_layerwise_from_gpu(
    connector, memory_objs, starts, ends, kvcaches, slot_mapping
):
    """Execute a layerwise connector's batched_from_gpu (full pipeline)."""
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
    """Execute a layerwise connector's batched_to_gpu (full pipeline)."""
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
# Benchmark: Layerwise XPU connector  (PyTorch index ops + XPU streams)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for GPU connector benchmarks",
)
def test_layerwise_xpu_connector_from_gpu_bench(benchmark, mock_xpu_as_cuda):
    """Benchmark VLLMPagedMemLayerwiseXPUConnector.batched_from_gpu
    (GPU → memory objs, layer by layer).

    Uses pure PyTorch index_select with XPU streams (mocked as CUDA
    streams for benchmarking).
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

    connector = VLLMPagedMemLayerwiseXPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=torch.device(DEVICE),
    )

    starts = [0]
    ends = [CHUNK_SIZE]

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
def test_layerwise_xpu_connector_to_gpu_bench(benchmark, mock_xpu_as_cuda):
    """Benchmark VLLMPagedMemLayerwiseXPUConnector.batched_to_gpu
    (memory objs → GPU, layer by layer).

    Uses pure PyTorch index_copy_ with XPU streams (mocked as CUDA
    streams for benchmarking).
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

    connector = VLLMPagedMemLayerwiseXPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=torch.device(DEVICE),
    )

    starts = [0]
    ends = [CHUNK_SIZE]

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
def test_layerwise_gpu_vs_xpu_connector_comparison(mock_xpu_as_cuda):
    """Side-by-side timing comparison of Layerwise GPU connector
    (CUDA kernels) vs Layerwise XPU connector (pure PyTorch ops).

    Both connectors follow the same layerwise generator protocol, making
    this an apples-to-apples comparison of the data transfer strategy.

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
    kvcaches_dst_gpu = generate_kv_cache_paged_list_tensors(
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
    dtype = kvcaches_src[0][0].dtype
    slot_mapping = _make_slot_mapping(NUM_BLOCKS, BLOCK_SIZE, NUM_TOKENS)

    starts = [0]
    ends = [CHUNK_SIZE]

    # ---- Layerwise GPU connector setup ----
    gpu_connector = VLLMPagedMemLayerwiseGPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=DEVICE,
    )
    gpu_shape = gpu_connector.get_shape(CHUNK_SIZE)
    gpu_mem_objs = [
        [pin_allocator.allocate(gpu_shape, dtype, fmt=MemoryFormat.KV_T2D)]
        for _ in range(NUM_LAYERS)
    ]

    # ---- Layerwise XPU connector setup ----
    xpu_connector = VLLMPagedMemLayerwiseXPUConnector(
        HIDDEN_DIM,
        NUM_LAYERS,
        use_gpu=True,
        chunk_size=CHUNK_SIZE,
        dtype=dtype,
        device=torch.device(DEVICE),
    )
    xpu_shape = xpu_connector.get_shape(CHUNK_SIZE)
    xpu_mem_objs = [
        [pin_allocator.allocate(xpu_shape, dtype, fmt=MemoryFormat.KV_T2D)]
        for _ in range(NUM_LAYERS)
    ]

    def _timed_cuda(fn, warmup: int, runs: int) -> float:
        """Return median GPU time in milliseconds."""
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
    gpu_from_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_from_gpu(
            gpu_connector, gpu_mem_objs, starts, ends, kvcaches_src, slot_mapping
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    xpu_from_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_from_gpu(
            xpu_connector, xpu_mem_objs, starts, ends, kvcaches_src, slot_mapping
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    # ---- to_gpu benchmarks ----
    # Populate memory objs first
    _run_layerwise_from_gpu(
        gpu_connector, gpu_mem_objs, starts, ends, kvcaches_src, slot_mapping
    )
    _run_layerwise_from_gpu(
        xpu_connector, xpu_mem_objs, starts, ends, kvcaches_src, slot_mapping
    )

    gpu_to_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_to_gpu(
            gpu_connector,
            gpu_mem_objs,
            starts,
            ends,
            kvcaches_dst_gpu,
            slot_mapping,
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    xpu_to_gpu_ms = _timed_cuda(
        lambda: _run_layerwise_to_gpu(
            xpu_connector,
            xpu_mem_objs,
            starts,
            ends,
            kvcaches_dst_xpu,
            slot_mapping,
        ),
        warmup=num_warmup,
        runs=num_runs,
    )

    # ---- Print comparison table ----
    header = (
        f"\n{'=' * 75}\n"
        f"  Performance Comparison: Layerwise GPU (CUDA) vs Layerwise XPU (PyTorch)\n"
        f"  Config: {NUM_LAYERS} layers, {NUM_HEADS} heads, "
        f"head_size={HEAD_SIZE}, chunk={CHUNK_SIZE} tokens\n"
        f"{'=' * 75}"
    )
    row_fmt = "  {:<35s} {:>12.3f} ms  {:>12.3f} ms  {:>8.2f}x"
    print(header)
    print(
        f"  {'Operation':<35s} {'GPU (CUDA)':>15s}  "
        f"{'XPU (PyTorch)':>15s}  {'Ratio':>8s}"
    )
    print(f"  {'-' * 35} {'-' * 15}  {'-' * 15}  {'-' * 8}")
    print(
        row_fmt.format(
            "from_gpu (device → host)",
            gpu_from_gpu_ms,
            xpu_from_gpu_ms,
            xpu_from_gpu_ms / gpu_from_gpu_ms if gpu_from_gpu_ms > 0 else float("inf"),
        )
    )
    print(
        row_fmt.format(
            "to_gpu (host → device)",
            gpu_to_gpu_ms,
            xpu_to_gpu_ms,
            xpu_to_gpu_ms / gpu_to_gpu_ms if gpu_to_gpu_ms > 0 else float("inf"),
        )
    )
    print(f"{'=' * 75}\n")

    # Cleanup
    for layer_objs_list in gpu_mem_objs:
        for obj in layer_objs_list:
            obj.ref_count_down()
    for layer_objs_list in xpu_mem_objs:
        for obj in layer_objs_list:
            obj.ref_count_down()

    assert pin_allocator.memcheck()
    assert gpu_connector.gpu_buffer_allocator.memcheck()
    assert xpu_connector.gpu_buffer_allocator.memcheck()
    pin_allocator.close()
