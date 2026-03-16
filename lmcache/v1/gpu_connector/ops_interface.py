# SPDX-License-Identifier: Apache-2.0
"""
Device-agnostic kernel operations interface for LMCache.

This module defines:
  - :class:`GPUKVFormat`: Python enumeration of supported GPU KV cache layouts.
  - :class:`TransferDirection`: Python enumeration of KV transfer directions.
  - :class:`KernelOpsInterface`: Abstract base class that every device backend
    must implement in order to participate in LMCache's KV-cache transfer pipeline.

By keeping the enum definitions in pure Python (rather than inside the CUDA
C-extension), utility functions such as :func:`discover_gpu_kv_format` can work
on any accelerator without requiring ``lmcache.c_ops`` to be installed.

To add support for a new device (e.g. HPU, IPU, …) a developer only needs to:

1. Subclass :class:`KernelOpsInterface`.
2. Implement every abstract method using the device-native APIs.
3. Register / instantiate the new backend in the appropriate connector factory.

The CUDA implementation lives in :mod:`lmcache.v1.gpu_connector.cuda_ops`.
"""

# Standard
import abc
from enum import IntEnum

# Third Party
import torch


class GPUKVFormat(IntEnum):
    """Enumeration of supported GPU KV cache memory layouts.

    Integer values are kept identical to the C++ ``GPUKVFormat`` enum defined
    in ``csrc/mem_kernels.cuh`` so that the two representations are always
    interchangeable.

    Symbol reference (mirrors the C++ comment block):
      - ``NL``   – number of layers
      - ``NB``   – number of blocks / pages
      - ``BS``   – block / page size
      - ``NBBS`` – page buffer size = NB × BS
      - ``NH``   – number of heads
      - ``HS``   – head size
      - ``TWO``  – the literal dimension ``2`` (for K and V)
      - ``ONE``  – the literal dimension ``1``
      - ``_X_``  – separates list-level dimensions
      - ``_``    – separates tensor dimensions within the same tensor
    """

    NB_NL_TWO_BS_NH_HS = 0
    """vLLM CROSS_LAYER mode.
    Layout: ``[num_blocks, num_layers, 2, block_size, num_heads, head_size]``
    """

    NL_X_TWO_NB_BS_NH_HS = 1
    """vLLM non-MLA flash attention.
    Layout: ``List[num_layers]`` of
    ``[2, num_blocks, block_size, num_heads, head_size]``
    """

    NL_X_NB_TWO_BS_NH_HS = 2
    """vLLM non-MLA flash infer.
    Layout: ``List[num_layers]`` of
    ``[num_blocks, 2, block_size, num_heads, head_size]``
    """

    NL_X_NB_BS_HS = 3
    """vLLM MLA (Multi-head Latent Attention).
    Layout: ``List[num_layers]`` of ``[num_blocks, block_size, head_size]``
    """

    TWO_X_NL_X_NBBS_NH_HS = 4
    """SGLang MHA (flash attention and flash infer).
    Layout: ``List[2]`` of ``List[num_layers]`` of
    ``[page_buffer_size, num_heads, head_size]``
    """

    NL_X_NBBS_ONE_HS = 5
    """SGLang MLA.
    Layout: ``List[num_layers]`` of ``[page_buffer_size, 1, head_size]``
    """


class TransferDirection(IntEnum):
    """Direction of a KV cache transfer operation.

    Integer values are kept identical to the C++ ``TransferDirection`` enum
    defined in ``csrc/mem_kernels.cuh``.
    """

    H2D = 0
    """Host to Device transfer."""

    D2H = 1
    """Device to Host transfer."""


class KernelOpsInterface(abc.ABC):
    """Abstract interface for device-specific KV cache kernel operations.

    All accelerator backends (CUDA, ROCm, XPU, HPU, …) must subclass this
    interface and implement every abstract method in order to support
    LMCache's KV cache transfer pipeline.

    The CUDA reference implementation is
    :class:`~lmcache.v1.gpu_connector.cuda_ops.CUDAKernelOps`.

    Notes
    -----
    - Methods that accept :class:`TransferDirection` or :class:`GPUKVFormat`
      arguments always receive the *Python* enum values defined in this module,
      never the C-extension enum objects.
    - Implementations are responsible for converting those values to whatever
      internal representation the device API requires.
    """

    @abc.abstractmethod
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
        """Transfer KV cache data between LMCache format and paged GPU memory
        across all model layers in a single call.

        Args:
            key_value: The LMCache-format KV cache tensor.
            key_value_ptrs: GPU tensor holding raw data pointers to each
                per-layer KV cache buffer.
            slot_mapping: 1-D tensor mapping token positions to paged-memory
                slot indices.
            paged_memory_device: The device on which the paged KV cache lives.
            page_buffer_size: Total number of slots in the paged KV buffer
                (``num_blocks * block_size``).
            direction: Transfer direction (H2D or D2H).
            gpu_kv_format: Layout of the paged GPU KV cache.
            block_size: Number of tokens per page block (0 = auto-detect from
                the format).
            skip_prefix_n_tokens: Number of leading tokens to skip writing;
                used to avoid overwriting APC-shared blocks.
        """
        raise NotImplementedError

    @abc.abstractmethod
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
        """Variant of :meth:`multi_layer_kv_transfer` used by SGLang.

        Collapses to the regular multi-layer transfer for MLA.

        Args:
            key_value: The LMCache-format KV cache tensor.
            key_value_ptrs: GPU tensor holding raw data pointers per layer.
            slot_mapping: Token-to-slot mapping tensor.
            paged_memory_device: Device for the paged KV cache.
            page_buffer_size: Total paged KV buffer slots.
            direction: Transfer direction (H2D or D2H).
            gpu_kv_format: Layout of the paged GPU KV cache.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def single_layer_kv_transfer(
        self,
        lmc_key_value_cache: torch.Tensor,
        vllm_key_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: TransferDirection,
        gpu_kv_format: GPUKVFormat,
        token_major: bool = False,
    ) -> None:
        """Transfer KV cache for a single model layer (vLLM variant).

        Args:
            lmc_key_value_cache: LMCache-format KV tensor for this layer.
            vllm_key_value_cache: vLLM paged KV tensor for this layer.
            slot_mapping: Token-to-slot mapping tensor.
            direction: Transfer direction (H2D or D2H).
            gpu_kv_format: Layout of the paged GPU KV cache.
            token_major: Whether ``lmc_key_value_cache`` is token-major.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def single_layer_kv_transfer_sgl(
        self,
        lmc_key_value_cache: torch.Tensor,
        sgl_key_cache: torch.Tensor,
        sgl_value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        direction: TransferDirection,
        token_major: bool = False,
    ) -> None:
        """Transfer KV cache for a single model layer (SGLang variant).

        SGLang stores K and V as separate tensors, hence the two cache
        arguments.

        Args:
            lmc_key_value_cache: LMCache-format KV tensor for this layer.
            sgl_key_cache: SGLang key-cache tensor for this layer.
            sgl_value_cache: SGLang value-cache tensor for this layer.
            slot_mapping: Token-to-slot mapping tensor.
            direction: Transfer direction (H2D or D2H).
            token_major: Whether ``lmc_key_value_cache`` is token-major.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def lmcache_memcpy_async(
        self,
        dst_ptr: int,
        src_ptr: int,
        nbytes: int,
        direction: TransferDirection,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        """Asynchronous memory copy between host and device.

        This call does *not* perform stream synchronisation; callers must
        synchronise themselves when required.

        Args:
            dst_ptr: Raw integer pointer to the destination buffer.
            src_ptr: Raw integer pointer to the source buffer.
            nbytes: Number of bytes to copy.
            direction: Transfer direction (H2D or D2H).
            host_buffer_offset: Byte offset into the pinned host buffer.
            host_buffer_alignments: Alignment of the pinned host buffer chunks
                (usually
                :attr:`~lmcache.v1.lazy_memory_allocator.LazyMemoryAllocator.PIN_CHUNK_SIZE`).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def rotary_embedding_k_fused(
        self,
        old_positions: torch.Tensor,
        new_positions: torch.Tensor,
        key: torch.Tensor,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> None:
        """Apply fused rotary position embeddings to a key-cache tensor
        in-place, rotating from ``old_positions`` to ``new_positions``.

        Args:
            old_positions: 1-D tensor with original token positions.
            new_positions: 1-D tensor with target token positions.
            key: Key tensor to update in-place;
                shape ``[num_tokens, num_heads, head_size]``.
            head_size: Size of each attention head.
            cos_sin_cache: Pre-computed cosine/sine cache tensor.
            is_neox_style: Whether to use NeoX-style positional encoding.
        """
        raise NotImplementedError
