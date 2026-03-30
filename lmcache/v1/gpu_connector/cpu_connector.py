# SPDX-License-Identifier: Apache-2.0
"""CPU KV connector for pure-CPU inference (aggregated / non-layerwise case).

This module provides a connector that transfers KV cache data between
paged CPU KV caches and contiguous :class:`MemoryObj` buffers using
pure PyTorch operations — no CUDA kernels or CUDA streams required.
"""

# Standard
from enum import Enum, auto
from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# CPU-compatible KV cache format enum (mirrors the GPU formats but without
# any dependency on lmc_ops / CUDA).
# ---------------------------------------------------------------------------
class CPUKVFormat(Enum):
    """Supported paged KV cache tensor layouts on CPU.

    Each value describes the per-layer tensor shape:

    * ``NL_X_TWO_NB_BS_NH_HS`` – ``[2, NB, BS, NH, HS]``
      (vLLM FlashAttention NHD)
    * ``NL_X_NB_TWO_BS_NH_HS`` – ``[NB, 2, BS, NH, HS]``
      (vLLM FlashInfer NHD)
    * ``NL_X_NB_BS_HS`` – ``[NB, BS, HS]``
      (vLLM MLA, no separate K/V split)
    """

    NL_X_TWO_NB_BS_NH_HS = auto()
    NL_X_NB_TWO_BS_NH_HS = auto()
    NL_X_NB_BS_HS = auto()


def discover_cpu_kv_format(kv_caches: List[torch.Tensor]) -> CPUKVFormat:
    """Detect the paged KV cache format from tensor shapes.

    Args:
        kv_caches: Per-layer KV cache tensors (``List[torch.Tensor]``).

    Returns:
        The detected :class:`CPUKVFormat`.

    Raises:
        ValueError: If the tensor shape does not match any known format.
    """
    t = kv_caches[0]
    ndim = t.ndim

    if ndim == 5:
        if t.shape[0] == 2:
            return CPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        if t.shape[1] == 2:
            return CPUKVFormat.NL_X_NB_TWO_BS_NH_HS
    elif ndim == 3:
        return CPUKVFormat.NL_X_NB_BS_HS

    raise ValueError(
        f"Unsupported CPU KV cache tensor shape {tuple(t.shape)} "
        f"(ndim={ndim}). Expected 5-d (non-MLA) or 3-d (MLA)."
    )


# ---------------------------------------------------------------------------
# Helpers for extracting KV format metadata from tensor shapes.
# ---------------------------------------------------------------------------


def _get_block_size(kv_caches: List[torch.Tensor], fmt: CPUKVFormat) -> int:
    """Return the block size for the given format."""
    t = kv_caches[0]
    if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        return t.shape[2]  # [2, NB, BS, NH, HS]
    if fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        return t.shape[2]  # [NB, 2, BS, NH, HS]
    if fmt == CPUKVFormat.NL_X_NB_BS_HS:
        return t.shape[1]  # [NB, BS, HS]
    raise ValueError(f"Unknown CPU KV format: {fmt}")


def _get_num_blocks(kv_caches: List[torch.Tensor], fmt: CPUKVFormat) -> int:
    """Return the number of blocks for the given format."""
    t = kv_caches[0]
    if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        return t.shape[1]  # [2, NB, BS, NH, HS]
    if fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        return t.shape[0]  # [NB, 2, BS, NH, HS]
    if fmt == CPUKVFormat.NL_X_NB_BS_HS:
        return t.shape[0]  # [NB, BS, HS]
    raise ValueError(f"Unknown CPU KV format: {fmt}")


def _get_hidden_dim(kv_caches: List[torch.Tensor], fmt: CPUKVFormat) -> int:
    """Return the hidden dimension (num_heads * head_size, or head_size for MLA)."""
    t = kv_caches[0]
    if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        return t.shape[3] * t.shape[4]  # NH * HS
    if fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        return t.shape[3] * t.shape[4]  # NH * HS
    if fmt == CPUKVFormat.NL_X_NB_BS_HS:
        return t.shape[2]  # HS
    raise ValueError(f"Unknown CPU KV format: {fmt}")


# ---------------------------------------------------------------------------
# Scatter / gather helpers (pure PyTorch, no CUDA).
# ---------------------------------------------------------------------------


def _gather_from_paged_kv(
    kv_layer: torch.Tensor,
    slot_indices: torch.Tensor,
    fmt: CPUKVFormat,
    block_size: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Gather contiguous KV data from a paged tensor using *slot_indices*.

    Args:
        kv_layer: Single-layer paged KV cache tensor.
        slot_indices: 1-D tensor of slot ids to gather.
        fmt: The format of the KV cache tensor.
        block_size: Block size of the paged cache.
        hidden_dim: Hidden dimension size.

    Returns:
        Tensor of shape ``[kv_size, num_tokens, hidden_dim]`` where
        *kv_size* is 2 for non-MLA formats and 1 for MLA.
    """
    block_indices = torch.div(slot_indices, block_size, rounding_mode="trunc")
    block_offsets = slot_indices % block_size

    if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # kv_layer: [2, NB, BS, NH, HS]
        k = kv_layer[0, block_indices, block_offsets].reshape(-1, hidden_dim)
        v = kv_layer[1, block_indices, block_offsets].reshape(-1, hidden_dim)
        return torch.stack([k, v], dim=0)  # [2, num_tokens, hidden_dim]

    if fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # kv_layer: [NB, 2, BS, NH, HS]
        k = kv_layer[block_indices, 0, block_offsets].reshape(-1, hidden_dim)
        v = kv_layer[block_indices, 1, block_offsets].reshape(-1, hidden_dim)
        return torch.stack([k, v], dim=0)  # [2, num_tokens, hidden_dim]

    if fmt == CPUKVFormat.NL_X_NB_BS_HS:
        # kv_layer: [NB, BS, HS]  (MLA – single combined tensor)
        data = kv_layer[block_indices, block_offsets]  # [num_tokens, HS]
        return data.unsqueeze(0)  # [1, num_tokens, hidden_dim]

    raise ValueError(f"Unknown CPU KV format: {fmt}")


def _scatter_to_paged_kv(
    kv_layer: torch.Tensor,
    data: torch.Tensor,
    slot_indices: torch.Tensor,
    fmt: CPUKVFormat,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> None:
    """Scatter contiguous KV data back into a paged tensor.

    Args:
        kv_layer: Single-layer paged KV cache tensor (modified in-place).
        data: Contiguous data of shape ``[kv_size, num_tokens, hidden_dim]``.
        slot_indices: 1-D tensor of slot ids.
        fmt: The format of the KV cache tensor.
        block_size: Block size of the paged cache.
        num_heads: Number of KV heads.
        head_size: Head dimension size.
    """
    block_indices = torch.div(slot_indices, block_size, rounding_mode="trunc")
    block_offsets = slot_indices % block_size
    num_tokens = slot_indices.shape[0]

    if fmt == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        # kv_layer: [2, NB, BS, NH, HS]
        kv_layer[0, block_indices, block_offsets] = data[0].reshape(
            num_tokens, num_heads, head_size
        )
        kv_layer[1, block_indices, block_offsets] = data[1].reshape(
            num_tokens, num_heads, head_size
        )
        return

    if fmt == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        # kv_layer: [NB, 2, BS, NH, HS]
        kv_layer[block_indices, 0, block_offsets] = data[0].reshape(
            num_tokens, num_heads, head_size
        )
        kv_layer[block_indices, 1, block_offsets] = data[1].reshape(
            num_tokens, num_heads, head_size
        )
        return

    if fmt == CPUKVFormat.NL_X_NB_BS_HS:
        # kv_layer: [NB, BS, HS]  (MLA)
        kv_layer[block_indices, block_offsets] = data[0]
        return

    raise ValueError(f"Unknown CPU KV format: {fmt}")


# ---------------------------------------------------------------------------
# CPU Connector
# ---------------------------------------------------------------------------


class VLLMPagedMemCPUConnector(GPUConnectorInterface):
    """CPU-native KV connector for the aggregated (non-layerwise) case.

    Transfers KV data between paged CPU KV caches and contiguous
    :class:`MemoryObj` buffers using PyTorch advanced indexing — no CUDA
    kernels, no CUDA streams.

    The connector supports the following vLLM KV cache formats:

    * ``NL_X_TWO_NB_BS_NH_HS`` – ``[2, NB, BS, NH, HS]``
      (FlashAttention NHD)
    * ``NL_X_NB_TWO_BS_NH_HS`` – ``[NB, 2, BS, NH, HS]``
      (FlashInfer NHD)
    * ``NL_X_NB_BS_HS`` – ``[NB, BS, HS]``
      (MLA)

    Args:
        hidden_dim_size: Hidden dimension of the KV cache
            (``num_heads * head_size``).
        num_layers: Number of transformer layers.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_mla: bool = kwargs.get("use_mla", False)

        # Filled lazily on first call to _initialize_format()
        self.cpu_kv_format: Optional[CPUKVFormat] = None
        self.block_size: int = 0
        self.num_blocks: int = 0
        self.num_heads: int = 0
        self.head_size: int = 0

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
    ) -> "VLLMPagedMemCPUConnector":
        """Create a connector from :class:`LMCacheMetadata`.

        Args:
            metadata: Model metadata containing KV shape information.

        Returns:
            A new ``VLLMPagedMemCPUConnector`` instance.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_mla=metadata.use_mla,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_format(self, kv_caches: List[torch.Tensor]) -> None:
        """Detect format and cache metadata from the KV cache tensors.

        Called lazily on the first ``to_gpu`` / ``from_gpu`` invocation.
        """
        if self.cpu_kv_format is not None:
            return

        self.cpu_kv_format = discover_cpu_kv_format(kv_caches)
        self.block_size = _get_block_size(kv_caches, self.cpu_kv_format)
        self.num_blocks = _get_num_blocks(kv_caches, self.cpu_kv_format)
        self.hidden_dim_size = _get_hidden_dim(kv_caches, self.cpu_kv_format)

        if self.cpu_kv_format == CPUKVFormat.NL_X_NB_BS_HS:
            self.num_heads = 1
            self.head_size = kv_caches[0].shape[2]
        elif self.cpu_kv_format == CPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
            self.num_heads = kv_caches[0].shape[3]
            self.head_size = kv_caches[0].shape[4]
        elif self.cpu_kv_format == CPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
            self.num_heads = kv_caches[0].shape[3]
            self.head_size = kv_caches[0].shape[4]

        logger.info(
            "CPU KV connector initialized: format=%s, block_size=%d, "
            "num_blocks=%d, hidden_dim=%d",
            self.cpu_kv_format.name,
            self.block_size,
            self.num_blocks,
            self.hidden_dim_size,
        )

    # ------------------------------------------------------------------
    # GPUConnectorInterface implementation
    # ------------------------------------------------------------------

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Copy data from a *MemoryObj* into the paged CPU KV caches.

        Despite the name (inherited from the GPU interface), this method
        operates entirely on CPU tensors.

        Required kwargs:
            kvcaches (List[torch.Tensor]): Per-layer paged KV cache tensors.
            slot_mapping (torch.Tensor): Full slot mapping for the sequence.

        Args:
            memory_obj: Source memory object (format ``KV_2LTD`` or
                ``KV_MLA_FMT``).
            start: Start index in the token sequence.
            end: End index in the token sequence.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches must be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "Expected KV_MLA_FMT format for MLA model, got "
                    f"{memory_obj.metadata.fmt}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    f"Expected KV_2LTD format, got {memory_obj.metadata.fmt}"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_format(self.kvcaches)
        assert self.cpu_kv_format is not None

        slot_indices = slot_mapping[start:end]

        # Handle prefix-caching skip
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip = min(end - start, max(0, vllm_cached - start))
        if skip > 0:
            slot_indices = slot_indices[skip:]

        data = memory_obj.tensor  # [kv_size, num_layers, num_tokens, hidden_dim]
        if skip > 0:
            data = data[:, :, skip:]

        for layer_idx in range(self.num_layers):
            _scatter_to_paged_kv(
                self.kvcaches[layer_idx],
                data[:, layer_idx],
                slot_indices,
                self.cpu_kv_format,
                self.block_size,
                self.num_heads,
                self.head_size,
            )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Copy data from the paged CPU KV caches into a *MemoryObj*.

        Despite the name (inherited from the GPU interface), this method
        operates entirely on CPU tensors.

        Required kwargs:
            kvcaches (List[torch.Tensor]): Per-layer paged KV cache tensors.
            slot_mapping (torch.Tensor): Full slot mapping for the sequence.

        Args:
            memory_obj: Target memory object; its ``metadata.fmt`` will be
                set to ``KV_2LTD`` (or ``KV_MLA_FMT`` for MLA models).
            start: Start index in the token sequence.
            end: End index in the token sequence.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches must be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' must be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self._initialize_format(self.kvcaches)
        assert self.cpu_kv_format is not None

        slot_indices = slot_mapping[start:end]

        for layer_idx in range(self.num_layers):
            gathered = _gather_from_paged_kv(
                self.kvcaches[layer_idx],
                slot_indices,
                self.cpu_kv_format,
                self.block_size,
                self.hidden_dim_size,
            )
            # gathered shape: [kv_size, num_tokens, hidden_dim]
            memory_obj.tensor[:, layer_idx, :, :] = gathered

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """Batched ``to_gpu`` for the aggregated (non-layerwise) case.

        Iterates over the given memory objects and writes each one into
        the paged CPU KV caches.

        Args:
            memory_objs: List of source memory objects.
            starts: Corresponding start indices.
            ends: Corresponding end indices.
        """
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        """Batched ``from_gpu`` for the aggregated (non-layerwise) case.

        Iterates over the given memory objects and reads data from the
        paged CPU KV caches.

        Args:
            memory_objs: List of target memory objects.
            starts: Corresponding start indices.
            ends: Corresponding end indices.
        """
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the memory-object tensor shape for *num_tokens*.

        Returns:
            ``torch.Size([kv_size, num_layers, num_tokens, hidden_dim])``
            where *kv_size* is 1 for MLA and 2 otherwise.
        """
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])
