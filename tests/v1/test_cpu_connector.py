# SPDX-License-Identifier: Apache-2.0
"""Tests for VLLMPagedMemCPUConnector (CPU-native aggregated KV connector)."""

# Standard
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.cpu_connector import (
    CPUKVFormat,
    VLLMPagedMemCPUConnector,
    discover_cpu_kv_format,
)
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_cpu_kv_caches(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    fmt: CPUKVFormat,
    dtype: torch.dtype = torch.bfloat16,
):
    """Create random paged KV cache tensors on CPU."""
    caches = []
    for _ in range(num_layers):
        if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
            shape = [2, num_blocks, block_size, num_heads, head_size]
        elif fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
            shape = [num_blocks, 2, block_size, num_heads, head_size]
        elif fmt == CPUKVFormat.NL_X_NB_BS_HS:
            shape = [num_blocks, block_size, head_size]
        else:
            raise ValueError(f"Unknown format: {fmt}")
        caches.append(torch.rand(shape, dtype=dtype, device="cpu"))
    return caches


def _check_paged_equal_at_slots(
    left,
    right,
    slot_mapping: torch.Tensor,
    fmt: CPUKVFormat,
    block_size: int,
    hidden_dim: int,
):
    """Assert that *left* and *right* paged KV caches agree at *slot_mapping*."""
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="trunc")
    block_offsets = slot_mapping % block_size

    for l_kv, r_kv in zip(left, right, strict=False):
        if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
            l_k = l_kv[0, block_indices, block_offsets].reshape(-1, hidden_dim)
            r_k = r_kv[0, block_indices, block_offsets].reshape(-1, hidden_dim)
            l_v = l_kv[1, block_indices, block_offsets].reshape(-1, hidden_dim)
            r_v = r_kv[1, block_indices, block_offsets].reshape(-1, hidden_dim)
            assert torch.equal(l_k, r_k), "K mismatch"
            assert torch.equal(l_v, r_v), "V mismatch"
        elif fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
            l_k = l_kv[block_indices, 0, block_offsets].reshape(-1, hidden_dim)
            r_k = r_kv[block_indices, 0, block_offsets].reshape(-1, hidden_dim)
            l_v = l_kv[block_indices, 1, block_offsets].reshape(-1, hidden_dim)
            r_v = r_kv[block_indices, 1, block_offsets].reshape(-1, hidden_dim)
            assert torch.equal(l_k, r_k), "K mismatch"
            assert torch.equal(l_v, r_v), "V mismatch"
        elif fmt == CPUKVFormat.NL_X_NB_BS_HS:
            l_d = l_kv[block_indices, block_offsets]
            r_d = r_kv[block_indices, block_offsets]
            assert torch.equal(l_d, r_d), "MLA data mismatch"


# ---------------------------------------------------------------------------
# Tests: format discovery
# ---------------------------------------------------------------------------


class TestDiscoverCPUKVFormat:
    """Tests for :func:`discover_cpu_kv_format`."""

    def test_flash_attention_nhd(self):
        t = torch.empty(2, 10, 16, 8, 128)
        assert discover_cpu_kv_format([t]) == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS

    def test_flash_infer_nhd(self):
        t = torch.empty(10, 2, 16, 8, 128)
        assert discover_cpu_kv_format([t]) == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS

    def test_mla(self):
        t = torch.empty(10, 16, 128)
        assert discover_cpu_kv_format([t]) == CPUKVFormat.NL_X_NB_BS_HS

    def test_unsupported_4d(self):
        t = torch.empty(10, 16, 8, 128)
        with pytest.raises(ValueError, match="Unsupported"):
            discover_cpu_kv_format([t])


# ---------------------------------------------------------------------------
# Tests: round-trip (from_gpu → to_gpu) for each format
# ---------------------------------------------------------------------------


NUM_LAYERS = 4
NUM_BLOCKS = 20
BLOCK_SIZE = 16
NUM_HEADS = 8
HEAD_SIZE = 128
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE
NUM_TOKENS = 100
CHUNK_SIZE = 64


@pytest.mark.parametrize(
    "fmt",
    [
        CPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        CPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    ],
)
def test_cpu_connector_roundtrip_non_mla(fmt):
    """from_gpu(src) -> to_gpu(dst): data at slot_mapping should match."""
    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), NUM_TOKENS),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()

    conn_src = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    conn_dst = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)

    for start in range(0, NUM_TOKENS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, NUM_TOKENS)
        shape = conn_src.get_shape(end - start)
        mem_obj = allocator.allocate(shape, torch.bfloat16)

        conn_src.from_gpu(
            mem_obj,
            start,
            end,
            kvcaches=src,
            slot_mapping=slot_mapping,
        )
        assert mem_obj.metadata.fmt == MemoryFormat.KV_2LTD

        conn_dst.to_gpu(
            mem_obj,
            start,
            end,
            kvcaches=dst,
            slot_mapping=slot_mapping,
        )
        allocator.free(mem_obj)

    _check_paged_equal_at_slots(src, dst, slot_mapping, fmt, BLOCK_SIZE, HIDDEN_DIM)


def test_cpu_connector_roundtrip_mla():
    """Round-trip test for MLA format (NL_X_NB_BS_HS)."""
    fmt = CPUKVFormat.NL_X_NB_BS_HS
    mla_head_size = 128
    mla_num_heads = 1
    mla_hidden = mla_head_size

    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, mla_num_heads, mla_head_size, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, mla_num_heads, mla_head_size, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), NUM_TOKENS),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()

    conn_src = VLLMPagedMemCPUConnector(mla_hidden, NUM_LAYERS, use_mla=True)
    conn_dst = VLLMPagedMemCPUConnector(mla_hidden, NUM_LAYERS, use_mla=True)

    for start in range(0, NUM_TOKENS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, NUM_TOKENS)
        shape = conn_src.get_shape(end - start)
        mem_obj = allocator.allocate(shape, torch.bfloat16)

        conn_src.from_gpu(
            mem_obj,
            start,
            end,
            kvcaches=src,
            slot_mapping=slot_mapping,
        )
        assert mem_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT

        conn_dst.to_gpu(
            mem_obj,
            start,
            end,
            kvcaches=dst,
            slot_mapping=slot_mapping,
        )
        allocator.free(mem_obj)

    _check_paged_equal_at_slots(src, dst, slot_mapping, fmt, BLOCK_SIZE, mla_hidden)


# ---------------------------------------------------------------------------
# Tests: batched operations
# ---------------------------------------------------------------------------


def test_cpu_connector_batched_roundtrip():
    """batched_from_gpu → batched_to_gpu round-trip."""
    fmt = CPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), NUM_TOKENS),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()

    conn_src = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    conn_dst = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)

    # Build chunks
    mem_objs = []
    starts = []
    ends = []
    for start in range(0, NUM_TOKENS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, NUM_TOKENS)
        shape = conn_src.get_shape(end - start)
        mem_objs.append(allocator.allocate(shape, torch.bfloat16))
        starts.append(start)
        ends.append(end)

    conn_src.batched_from_gpu(
        mem_objs,
        starts,
        ends,
        kvcaches=src,
        slot_mapping=slot_mapping,
    )
    conn_dst.batched_to_gpu(
        mem_objs,
        starts,
        ends,
        kvcaches=dst,
        slot_mapping=slot_mapping,
    )

    _check_paged_equal_at_slots(src, dst, slot_mapping, fmt, BLOCK_SIZE, HIDDEN_DIM)

    for m in mem_objs:
        allocator.free(m)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


def test_cpu_connector_missing_slot_mapping():
    """Should raise ValueError when slot_mapping is missing."""
    fmt = CPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    kv = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    allocator = AdHocMemoryAllocator()
    conn = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    shape = conn.get_shape(10)
    mem_obj = allocator.allocate(shape, torch.bfloat16)

    with pytest.raises(ValueError, match="slot_mapping"):
        conn.from_gpu(mem_obj, 0, 10, kvcaches=kv)

    with pytest.raises(ValueError, match="slot_mapping"):
        conn.to_gpu(mem_obj, 0, 10, kvcaches=kv)

    allocator.free(mem_obj)


def test_cpu_connector_wrong_format_for_to_gpu():
    """to_gpu should reject memory objects with wrong format."""
    fmt = CPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    kv = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), 10),
        dtype=torch.int64,
    )
    allocator = AdHocMemoryAllocator()

    # Non-MLA connector expects KV_2LTD
    conn = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    shape = conn.get_shape(10)
    mem_obj = allocator.allocate(shape, torch.bfloat16)
    mem_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT  # wrong format

    with pytest.raises(ValueError, match="KV_2LTD"):
        conn.to_gpu(
            mem_obj,
            0,
            10,
            kvcaches=kv,
            slot_mapping=slot_mapping,
        )

    allocator.free(mem_obj)


# ---------------------------------------------------------------------------
# Tests: get_shape
# ---------------------------------------------------------------------------


def test_cpu_connector_get_shape():
    conn = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    shape = conn.get_shape(50)
    assert shape == torch.Size([2, NUM_LAYERS, 50, HIDDEN_DIM])


def test_cpu_connector_get_shape_mla():
    conn = VLLMPagedMemCPUConnector(128, NUM_LAYERS, use_mla=True)
    shape = conn.get_shape(50)
    assert shape == torch.Size([1, NUM_LAYERS, 50, 128])


# ---------------------------------------------------------------------------
# Tests: prefix caching (skip_prefix_n_tokens)
# ---------------------------------------------------------------------------


def test_cpu_connector_skip_prefix():
    """Verify that vllm_cached_tokens skips prefix tokens in to_gpu."""
    fmt = CPUKVFormat.NL_X_TWO_NB_BS_NH_HS
    num_tokens = 32

    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), num_tokens),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()
    conn_src = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    conn_dst = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)

    # from_gpu from source
    shape = conn_src.get_shape(num_tokens)
    mem_obj = allocator.allocate(shape, torch.bfloat16)
    conn_src.from_gpu(
        mem_obj,
        0,
        num_tokens,
        kvcaches=src,
        slot_mapping=slot_mapping,
    )

    # to_gpu with 16 tokens already cached (skip first 16)
    conn_dst.to_gpu(
        mem_obj,
        0,
        num_tokens,
        kvcaches=dst,
        slot_mapping=slot_mapping,
        vllm_cached_tokens=16,
    )

    # Only tokens 16..31 should match
    _check_paged_equal_at_slots(
        src, dst, slot_mapping[16:], fmt, BLOCK_SIZE, HIDDEN_DIM
    )

    allocator.free(mem_obj)


# ---------------------------------------------------------------------------
# Tests: end-to-end disk offloading
# CPU KV cache → MemoryObj → disk → MemoryObj → CPU KV cache
# ---------------------------------------------------------------------------


def test_cpu_connector_disk_offload_roundtrip(tmp_path):
    """End-to-end: CPU KV cache → MemoryObj → disk file → MemoryObj → CPU KV cache.

    This validates that the full pipeline works on CPU: the connector
    serialises paged KV data into a MemoryObj, the MemoryObj is written
    to disk as raw bytes, then read back, and the connector restores
    the data into a fresh paged KV cache that matches the original.
    """
    fmt = CPUKVFormat.NL_X_TWO_NB_BS_NH_HS

    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), NUM_TOKENS),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()
    conn_src = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)
    conn_dst = VLLMPagedMemCPUConnector(HIDDEN_DIM, NUM_LAYERS)

    for start in range(0, NUM_TOKENS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, NUM_TOKENS)
        shape = conn_src.get_shape(end - start)

        # Step 1: CPU KV cache → MemoryObj (via connector)
        mem_obj = allocator.allocate(shape, torch.bfloat16)
        conn_src.from_gpu(mem_obj, start, end, kvcaches=src, slot_mapping=slot_mapping)

        # Step 2: MemoryObj → disk (write raw bytes)
        disk_path = tmp_path / f"chunk_{start}_{end}.bin"
        buf = mem_obj.byte_array
        with open(disk_path, "wb") as f:
            f.write(buf)
        saved_shape = mem_obj.metadata.shape
        saved_dtype = mem_obj.metadata.dtype
        saved_fmt = mem_obj.metadata.fmt
        allocator.free(mem_obj)

        # Step 3: disk → MemoryObj (read raw bytes)
        mem_obj2 = allocator.allocate(saved_shape, saved_dtype)
        mem_obj2.metadata.fmt = saved_fmt
        buf2 = mem_obj2.byte_array
        with open(disk_path, "rb") as f:
            f.readinto(buf2)

        # Step 4: MemoryObj → CPU KV cache (via connector)
        conn_dst.to_gpu(mem_obj2, start, end, kvcaches=dst, slot_mapping=slot_mapping)
        allocator.free(mem_obj2)

    # Verify the round-trip matches
    _check_paged_equal_at_slots(src, dst, slot_mapping, fmt, BLOCK_SIZE, HIDDEN_DIM)


def test_cpu_connector_disk_offload_roundtrip_mla(tmp_path):
    """End-to-end disk offload round-trip for MLA format."""
    fmt = CPUKVFormat.NL_X_NB_BS_HS
    mla_head_size = 128
    mla_num_heads = 1
    mla_hidden = mla_head_size

    src = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, mla_num_heads, mla_head_size, fmt
    )
    dst = _generate_cpu_kv_caches(
        NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, mla_num_heads, mla_head_size, fmt
    )

    slot_mapping = torch.tensor(
        random.sample(range(NUM_BLOCKS * BLOCK_SIZE), NUM_TOKENS),
        dtype=torch.int64,
    )

    allocator = AdHocMemoryAllocator()
    conn_src = VLLMPagedMemCPUConnector(mla_hidden, NUM_LAYERS, use_mla=True)
    conn_dst = VLLMPagedMemCPUConnector(mla_hidden, NUM_LAYERS, use_mla=True)

    for start in range(0, NUM_TOKENS, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, NUM_TOKENS)
        shape = conn_src.get_shape(end - start)

        # CPU KV cache → MemoryObj
        mem_obj = allocator.allocate(shape, torch.bfloat16)
        conn_src.from_gpu(mem_obj, start, end, kvcaches=src, slot_mapping=slot_mapping)
        assert mem_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT

        # MemoryObj → disk
        disk_path = tmp_path / f"mla_chunk_{start}_{end}.bin"
        with open(disk_path, "wb") as f:
            f.write(mem_obj.byte_array)
        saved_shape = mem_obj.metadata.shape
        saved_dtype = mem_obj.metadata.dtype
        saved_fmt = mem_obj.metadata.fmt
        allocator.free(mem_obj)

        # disk → MemoryObj
        mem_obj2 = allocator.allocate(saved_shape, saved_dtype)
        mem_obj2.metadata.fmt = saved_fmt
        with open(disk_path, "rb") as f:
            f.readinto(mem_obj2.byte_array)

        # MemoryObj → CPU KV cache
        conn_dst.to_gpu(mem_obj2, start, end, kvcaches=dst, slot_mapping=slot_mapping)
        allocator.free(mem_obj2)

    _check_paged_equal_at_slots(src, dst, slot_mapping, fmt, BLOCK_SIZE, mla_hidden)
