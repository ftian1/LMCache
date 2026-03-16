# SPDX-License-Identifier: Apache-2.0
"""
Tests for the device-agnostic kernel operations abstraction layer.

These tests verify:
1. Python-native GPUKVFormat and TransferDirection enums are importable and
   have correct integer values (matching the C++ enum definitions in
   csrc/mem_kernels.cuh).
2. KernelOpsInterface is a proper abstract base class.
3. Utility functions (discover_gpu_kv_format, get_num_blocks, etc.) work with
   the Python enums without requiring lmcache.c_ops to be installed.
"""

# Standard
from typing import Any, List

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.ops_interface import (
    GPUKVFormat,
    KernelOpsInterface,
    TransferDirection,
)
from lmcache.v1.gpu_connector.utils import (
    discover_gpu_kv_format,
    get_block_size,
    get_elements_per_layer,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    get_tokens_per_layer,
    is_mla,
)


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


class TestGPUKVFormatEnum:
    """Verify GPUKVFormat values match the C++ enum in csrc/mem_kernels.cuh."""

    def test_integer_values_match_cpp_enum(self):
        """Integer values must be stable across Python / C++ boundary."""
        assert int(GPUKVFormat.NB_NL_TWO_BS_NH_HS) == 0
        assert int(GPUKVFormat.NL_X_TWO_NB_BS_NH_HS) == 1
        assert int(GPUKVFormat.NL_X_NB_TWO_BS_NH_HS) == 2
        assert int(GPUKVFormat.NL_X_NB_BS_HS) == 3
        assert int(GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS) == 4
        assert int(GPUKVFormat.NL_X_NBBS_ONE_HS) == 5

    def test_all_six_formats_exist(self):
        assert len(GPUKVFormat) == 6

    def test_is_int_enum(self):
        """GPUKVFormat must behave as int for comparisons."""
        assert GPUKVFormat.NL_X_TWO_NB_BS_NH_HS == 1
        assert GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS > GPUKVFormat.NL_X_NB_BS_HS


class TestTransferDirectionEnum:
    """Verify TransferDirection values match the C++ enum."""

    def test_integer_values_match_cpp_enum(self):
        assert int(TransferDirection.H2D) == 0
        assert int(TransferDirection.D2H) == 1

    def test_both_directions_exist(self):
        assert len(TransferDirection) == 2

    def test_is_int_enum(self):
        assert TransferDirection.H2D == 0
        assert TransferDirection.D2H == 1
        assert TransferDirection.D2H != TransferDirection.H2D


# ---------------------------------------------------------------------------
# KernelOpsInterface abstract-class enforcement
# ---------------------------------------------------------------------------


class TestKernelOpsInterface:
    """KernelOpsInterface must be a proper ABC that can be subclassed."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            KernelOpsInterface()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        """A concrete subclass missing any abstract method should raise."""

        class Incomplete(KernelOpsInterface):
            # Only implement one of many required methods
            def multi_layer_kv_transfer(self, *args, **kwargs):  # type: ignore[override]
                pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        """A subclass that implements every abstract method should work."""

        class NoopKernelOps(KernelOpsInterface):
            def multi_layer_kv_transfer(self, *args, **kwargs):
                pass

            def multi_layer_kv_transfer_unilateral(self, *args, **kwargs):
                pass

            def single_layer_kv_transfer(self, *args, **kwargs):
                pass

            def single_layer_kv_transfer_sgl(self, *args, **kwargs):
                pass

            def lmcache_memcpy_async(self, *args, **kwargs):
                pass

            def rotary_embedding_k_fused(self, *args, **kwargs):
                pass

        ops = NoopKernelOps()
        assert isinstance(ops, KernelOpsInterface)


# ---------------------------------------------------------------------------
# Utility-function tests (no GPU / c_ops required)
# ---------------------------------------------------------------------------


def _make_vllm_flash_attn_kv(
    num_layers: int = 4,
    num_blocks: int = 8,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 64,
) -> List[torch.Tensor]:
    """NL_X_TWO_NB_BS_NH_HS: List[NL] of [2, NB, BS, NH, HS]."""
    return [
        torch.zeros(2, num_blocks, block_size, num_heads, head_size)
        for _ in range(num_layers)
    ]


def _make_vllm_flash_infer_kv(
    num_layers: int = 4,
    num_blocks: int = 8,
    block_size: int = 16,
    num_heads: int = 8,
    head_size: int = 64,
) -> List[torch.Tensor]:
    """NL_X_NB_TWO_BS_NH_HS: List[NL] of [NB, 2, BS, NH, HS]."""
    return [
        torch.zeros(num_blocks, 2, block_size, num_heads, head_size)
        for _ in range(num_layers)
    ]


def _make_vllm_mla_kv(
    num_layers: int = 4,
    num_blocks: int = 8,
    block_size: int = 16,
    head_size: int = 512,
) -> List[torch.Tensor]:
    """NL_X_NB_BS_HS: List[NL] of [NB, BS, HS]."""
    return [torch.zeros(num_blocks, block_size, head_size) for _ in range(num_layers)]


def _make_sglang_mha_kv(
    num_layers: int = 4,
    page_buffer_size: int = 128,
    num_heads: int = 8,
    head_size: int = 64,
) -> List[List[torch.Tensor]]:
    """TWO_X_NL_X_NBBS_NH_HS: List[2] of List[NL] of [NBBS, NH, HS]."""
    k_list = [
        torch.zeros(page_buffer_size, num_heads, head_size) for _ in range(num_layers)
    ]
    v_list = [
        torch.zeros(page_buffer_size, num_heads, head_size) for _ in range(num_layers)
    ]
    return [k_list, v_list]


def _make_sglang_mla_kv(
    num_layers: int = 4,
    page_buffer_size: int = 128,
    head_size: int = 512,
) -> List[torch.Tensor]:
    """NL_X_NBBS_ONE_HS: List[NL] of [NBBS, 1, HS]."""
    return [torch.zeros(page_buffer_size, 1, head_size) for _ in range(num_layers)]


class TestDiscoverGPUKVFormat:
    """discover_gpu_kv_format must return Python GPUKVFormat enum values."""

    def test_vllm_flash_attn(self):
        kv = _make_vllm_flash_attn_kv()
        fmt = discover_gpu_kv_format(kv, EngineType.VLLM)
        assert fmt == GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert isinstance(fmt, GPUKVFormat)

    def test_vllm_flash_infer(self):
        kv = _make_vllm_flash_infer_kv()
        fmt = discover_gpu_kv_format(kv, EngineType.VLLM)
        assert fmt == GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
        assert isinstance(fmt, GPUKVFormat)

    def test_vllm_mla(self):
        kv = _make_vllm_mla_kv()
        fmt = discover_gpu_kv_format(kv, EngineType.VLLM)
        assert fmt == GPUKVFormat.NL_X_NB_BS_HS
        assert isinstance(fmt, GPUKVFormat)

    def test_sglang_mha(self):
        kv = _make_sglang_mha_kv()
        fmt = discover_gpu_kv_format(kv, EngineType.SGLANG)
        assert fmt == GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS
        assert isinstance(fmt, GPUKVFormat)

    def test_sglang_mla(self):
        kv = _make_sglang_mla_kv()
        fmt = discover_gpu_kv_format(kv, EngineType.SGLANG)
        assert fmt == GPUKVFormat.NL_X_NBBS_ONE_HS
        assert isinstance(fmt, GPUKVFormat)

    def test_unsupported_format_raises(self):
        # 4-D single tensor is not a recognised vLLM cross-layer format
        # (wrong list depth for cross-layer format which has list_depth=0)
        kv: Any = [[torch.zeros(3, 3, 3, 3), torch.zeros(3, 3, 3, 3)]]
        with pytest.raises(ValueError):
            discover_gpu_kv_format(kv, EngineType.VLLM)


class TestUtilFunctionsNoCUDA:
    """All geometry helpers must work with Python GPUKVFormat without c_ops."""

    NL, NB, BS, NH, HS = 4, 8, 16, 8, 64

    # ---- vLLM flash attention ----

    def test_get_num_layers_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_num_layers(kv, fmt) == self.NL

    def test_get_num_blocks_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_num_blocks(kv, fmt) == self.NB

    def test_get_block_size_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_block_size(kv, fmt) == self.BS

    def test_get_hidden_dim_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_hidden_dim_size(kv, fmt) == self.NH * self.HS

    def test_get_num_heads_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_num_heads(kv, fmt) == self.NH

    def test_get_head_size_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_head_size(kv, fmt) == self.HS

    def test_get_page_buffer_size_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_page_buffer_size(kv, fmt) == self.NB * self.BS

    def test_get_tokens_per_layer_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        assert get_tokens_per_layer(kv, fmt) == self.NB * self.BS

    def test_get_elements_per_layer_flash_attn(self):
        kv = _make_vllm_flash_attn_kv(self.NL, self.NB, self.BS, self.NH, self.HS)
        fmt = GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        # 2 * NB * BS * NH * HS
        expected = 2 * self.NB * self.BS * self.NH * self.HS
        assert get_elements_per_layer(kv, fmt) == expected

    # ---- vLLM MLA ----

    def test_is_mla_returns_true_for_vllm_mla(self):
        assert is_mla(GPUKVFormat.NL_X_NB_BS_HS)

    def test_is_mla_returns_false_for_flash_attn(self):
        assert not is_mla(GPUKVFormat.NL_X_TWO_NB_BS_NH_HS)

    # ---- SGLang MHA ----

    PBS = 128

    def test_get_page_buffer_size_sglang_mha(self):
        kv = _make_sglang_mha_kv(self.NL, self.PBS, self.NH, self.HS)
        fmt = GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS
        assert get_page_buffer_size(kv, fmt) == self.PBS

    def test_get_num_heads_sglang_mha(self):
        kv = _make_sglang_mha_kv(self.NL, self.PBS, self.NH, self.HS)
        fmt = GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS
        assert get_num_heads(kv, fmt) == self.NH
