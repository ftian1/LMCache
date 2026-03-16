# SPDX-License-Identifier: Apache-2.0
"""
CUDA implementation of
:class:`~lmcache.v1.gpu_connector.ops_interface.KernelOpsInterface`.

This module provides :class:`CUDAKernelOps`, which delegates every kernel
call to the ``lmcache.c_ops`` C-extension, converting Python-level
:class:`~lmcache.v1.gpu_connector.ops_interface.GPUKVFormat` and
:class:`~lmcache.v1.gpu_connector.ops_interface.TransferDirection` enum values
into their pybind11-bound C++ counterparts along the way.
"""

# Third Party
import torch

# First Party
import lmcache.c_ops as _c_ops
from lmcache.v1.gpu_connector.ops_interface import (
    GPUKVFormat,
    KernelOpsInterface,
    TransferDirection,
)

# ---------------------------------------------------------------------------
# Build explicit mapping tables at import time so that every kernel call pays
# only a single dict look-up per enum argument rather than an attribute lookup
# chain on the c_ops module.
# ---------------------------------------------------------------------------
_DIRECTION_TO_C: dict[TransferDirection, object] = {
    TransferDirection.H2D: _c_ops.TransferDirection.H2D,
    TransferDirection.D2H: _c_ops.TransferDirection.D2H,
}

_FORMAT_TO_C: dict[GPUKVFormat, object] = {
    GPUKVFormat.NB_NL_TWO_BS_NH_HS: _c_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,
    GPUKVFormat.NL_X_TWO_NB_BS_NH_HS: _c_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
    GPUKVFormat.NL_X_NB_TWO_BS_NH_HS: _c_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
    GPUKVFormat.NL_X_NB_BS_HS: _c_ops.GPUKVFormat.NL_X_NB_BS_HS,
    GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS: _c_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS,
    GPUKVFormat.NL_X_NBBS_ONE_HS: _c_ops.GPUKVFormat.NL_X_NBBS_ONE_HS,
}


class CUDAKernelOps(KernelOpsInterface):
    """CUDA backend for LMCache kernel operations.

    All methods simply delegate to the corresponding function in
    ``lmcache.c_ops``, converting the Python-native
    :class:`~lmcache.v1.gpu_connector.ops_interface.GPUKVFormat` /
    :class:`~lmcache.v1.gpu_connector.ops_interface.TransferDirection` enum
    values into the pybind11-bound C++ enum objects expected by the extension.

    Raises:
        ImportError: If ``lmcache.c_ops`` is not available (i.e. the package
            was installed without CUDA extension support).
    """

    def multi_layer_kv_transfer(
        self,
        key_value: torch.Tensor,
        key_value_ptrs: torch.Tensor,
        slot_mapping: torch.Tensor,
        paged_memory_device: torch.device,
        page_buffer_size: int,
        direction: TransferDirection,
        gpu_kv_format: GPUKVFormat,
        block_size: int = 0,
        skip_prefix_n_tokens: int = 0,
    ) -> None:
        """Delegate to ``lmcache.c_ops.multi_layer_kv_transfer``."""
        _c_ops.multi_layer_kv_transfer(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            _DIRECTION_TO_C[direction],
            _FORMAT_TO_C[gpu_kv_format],
            block_size,
            skip_prefix_n_tokens,
        )

    def multi_layer_kv_transfer_unilateral(
        self,
        key_value: torch.Tensor,
        key_value_ptrs: torch.Tensor,
        slot_mapping: torch.Tensor,
        paged_memory_device: torch.device,
        page_buffer_size: int,
        direction: TransferDirection,
        gpu_kv_format: GPUKVFormat,
    ) -> None:
        """Delegate to ``lmcache.c_ops.multi_layer_kv_transfer_unilateral``."""
        _c_ops.multi_layer_kv_transfer_unilateral(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            _DIRECTION_TO_C[direction],
            _FORMAT_TO_C[gpu_kv_format],
        )

    def single_layer_kv_transfer(
        self,
        lmc_key_value_cache: torch.Tensor,
        vllm_key_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: TransferDirection,
        gpu_kv_format: GPUKVFormat,
        token_major: bool = False,
    ) -> None:
        """Delegate to ``lmcache.c_ops.single_layer_kv_transfer``."""
        _c_ops.single_layer_kv_transfer(
            lmc_key_value_cache,
            vllm_key_value_cache,
            slot_mapping,
            _DIRECTION_TO_C[direction],
            _FORMAT_TO_C[gpu_kv_format],
            token_major,
        )

    def single_layer_kv_transfer_sgl(
        self,
        lmc_key_value_cache: torch.Tensor,
        sgl_key_cache: torch.Tensor,
        sgl_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: TransferDirection,
        token_major: bool = False,
    ) -> None:
        """Delegate to ``lmcache.c_ops.single_layer_kv_transfer_sgl``."""
        _c_ops.single_layer_kv_transfer_sgl(
            lmc_key_value_cache,
            sgl_key_cache,
            sgl_value_cache,
            slot_mapping,
            _DIRECTION_TO_C[direction],
            token_major,
        )

    def lmcache_memcpy_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        nbytes: int,
        direction: TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Delegate to ``lmcache.c_ops.lmcache_memcpy_async``."""
        _c_ops.lmcache_memcpy_async(
            dst_ptr,
            src_ptr,
            nbytes,
            _DIRECTION_TO_C[direction],
            host_buffer_offset,
            host_buffer_alignments,
        )

    def rotary_embedding_k_fused(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> None:
        """Delegate to ``lmcache.c_ops.rotary_embedding_k_fused``."""
        _c_ops.rotary_embedding_k_fused(
            old_positions,
            new_positions,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
        )
