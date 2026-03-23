# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Standard
from typing import List, Optional
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import _get_head_size_view, _split_token2d_kv
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)

# Valid format transitions for memory objects in the layerwise XPU connector.
# Only transitions to KV_MLA_FMT are allowed because the XPU layerwise
# connector produces MLA-formatted outputs when use_mla=True.  Transitions
# to KV_T2D happen implicitly via the allocator, not via the connector.
ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


class VLLMPagedMemXPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemXPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemXPUConnector"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            for i, kvcache in enumerate(kvcaches):
                kvcache.view(total_blocks, head_size).index_copy_(0, slices, tmp[i])
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            for i, (kcache, vcache) in enumerate(kvcaches):
                kcache.view(total_blocks, d).index_copy_(0, slices, tmp_k[i])
                vcache.view(total_blocks, d).index_copy_(0, slices, tmp_v[i])

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]

        if self.use_mla:
            num_blocks, block_size, head_size = kvcaches[0].shape
            total_blocks = num_blocks * block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, head_size).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
        else:
            num_blocks, block_size, num_heads, head_size = kvcaches[0][0].shape
            total_blocks = num_blocks * block_size
            d = num_heads * head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        if not memory_obj.tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.xpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)


class VLLMPagedMemLayerwiseXPUConnector(GPUConnectorInterface):
    """Layerwise paged KV connector for XPU.

    Implements the *same generator contract* as
    ``VLLMPagedMemLayerwiseGPUConnector``:

    - ``batched_to_gpu(...)`` yields ``num_layers + 2`` times
    - ``batched_from_gpu(...)`` yields ``num_layers + 1`` times

    Transfer is implemented with pure torch ops
    (``index_copy_`` / ``index_select``).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # XPU streams
        self.load_stream = torch.xpu.Stream()
        self.store_stream = torch.xpu.Stream()

        # Optional device staging buffer allocator (same pattern as CUDA)
        self.gpu_buffer_allocator = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseXPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model
                configuration.
            use_gpu: Whether to use a device staging buffer.
            device: The device to use for the connector.

        Returns:
            A new ``VLLMPagedMemLayerwiseXPUConnector`` instance.
        """
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(self, mem: MemoryObj, target_fmt):
        """Validate a memory format transition is allowed."""
        current_fmt = mem.metadata.fmt
        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        """Lazily initialize the XPU staging buffer allocator."""
        if self.use_gpu and self.gpu_buffer_allocator is None:
            # First Party
            from lmcache.v1.memory_management import GPUMemoryAllocator

            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )
            logger.info(
                "Initializing XPU staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                size=staging_bytes,
                device=self.device,
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Not supported — use ``batched_to_gpu`` (generator)."""
        raise NotImplementedError("Layerwise uses batched_to_gpu (generator).")

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Not supported — use ``batched_from_gpu`` (generator)."""
        raise NotImplementedError("Layerwise uses batched_from_gpu (generator).")

    def _batched_to_gpu_gen(self, starts: List[int], ends: List[int], **kwargs):
        """Generator: CPU token2d → (optional staging) → XPU paged KV.

        Yields ``num_layers + 2`` times (matching the CUDA layerwise
        connector contract).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        def _ensure_device(t: torch.Tensor) -> torch.Tensor:
            if t is None:
                return t
            if t.device != self.device:
                return t.to(self.device, non_blocking=True)
            return t

        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        slot_mapping_full = _ensure_device(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.xpu.current_stream().wait_stream(self.load_stream)
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate XPU staging buffer"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.xpu.current_stream()

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.xpu.stream(self.load_stream):
                    dst_layer = self.kvcaches[layer_id]
                    if self.use_mla:
                        dst_flat = _get_head_size_view(dst_layer, use_mla=True)
                    else:
                        dst_k_flat, dst_v_flat = _get_head_size_view(
                            dst_layer, use_mla=False
                        )

                    if self.use_gpu:
                        staged = tmp_gpu_buffer_obj.tensor
                        cursor = 0
                        for s, e, mem in zip(
                            starts,
                            ends,
                            memory_objs_layer,
                            strict=False,
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            assert cursor + n <= staged.shape[0], (
                                f"Staging buffer overflow: cursor={cursor}, "
                                f"n={n}, buffer_size={staged.shape[0]}"
                            )
                            src = _ensure_device(mem.tensor)
                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = _ensure_device(slot_mapping_full)
                        if self.use_mla:
                            staged_dev = _ensure_device(staged)
                            if staged_dev.dim() == 2:
                                dst_flat.index_copy_(0, sl, staged_dev)
                            elif staged_dev.dim() == 3 and staged_dev.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, staged_dev[0])
                            else:
                                raise ValueError(
                                    f"Unexpected MLA staged tensor: {staged_dev.shape}"
                                )
                        else:
                            k_tok, v_tok = _split_token2d_kv(staged)
                            k_tok = _ensure_device(k_tok)
                            v_tok = _ensure_device(v_tok)
                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)
                    else:
                        cursor = 0
                        for s, e, mem in zip(
                            starts,
                            ends,
                            memory_objs_layer,
                            strict=False,
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue
                            src = _ensure_device(mem.tensor)
                            sl = _ensure_device(slot_mapping_full[cursor : cursor + n])
                            cursor += n

                            if self.use_mla:
                                if src.dim() == 2:
                                    dst_flat.index_copy_(0, sl, src)
                                elif src.dim() == 3 and src.shape[0] == 1:
                                    dst_flat.index_copy_(0, sl, src[0])
                                else:
                                    raise ValueError(
                                        f"Unexpected MLA token tensor: {src.shape}"
                                    )
                            else:
                                k_tok, v_tok = _split_token2d_kv(src)
                                k_tok = _ensure_device(k_tok)
                                v_tok = _ensure_device(v_tok)
                                dst_k_flat.index_copy_(0, sl, k_tok)
                                dst_v_flat.index_copy_(0, sl, v_tok)

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """Generator: XPU paged KV → (optional staging) → CPU token2d.

        Yields ``num_layers + 1`` times (matching the CUDA layerwise
        connector contract).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.xpu.current_stream()

        def _copy_kv_into_mem(
            mem_tensor: torch.Tensor,
            k_src: torch.Tensor,
            v_src: torch.Tensor,
        ) -> None:
            """Copy K/V into mem.tensor (supports [2,...] or [...,2,...])."""
            if mem_tensor.dim() < 3:
                raise ValueError(
                    f"Unexpected output token2d layout: {mem_tensor.shape}"
                )
            if mem_tensor.shape[0] == 2:
                mem_tensor[0].copy_(k_src.to(mem_tensor[0].device), non_blocking=True)
                mem_tensor[1].copy_(v_src.to(mem_tensor[1].device), non_blocking=True)
                return
            if mem_tensor.shape[1] == 2:
                mem_tensor[:, 0, ...].copy_(
                    k_src.to(mem_tensor.device), non_blocking=True
                )
                mem_tensor[:, 1, ...].copy_(
                    v_src.to(mem_tensor.device), non_blocking=True
                )
                return
            raise ValueError(f"Unexpected output token2d layout: {mem_tensor.shape}")

        slot_mapping_on_device = slot_mapping.to(self.device)

        slot_mapping_full = torch.cat(
            [slot_mapping_on_device[s:e] for s, e in zip(starts, ends, strict=False)],
            dim=0,
        )
        total_tokens = int(slot_mapping_full.numel())

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(total_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate XPU staging buffer"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        try:
            for layer_id in range(self.num_layers):
                mem_layer = memory_objs[layer_id]

                with torch.xpu.stream(self.store_stream):
                    self.store_stream.wait_stream(current_stream)

                    src_layer = self.kvcaches[layer_id]

                    if self.use_mla:
                        src_flat = _get_head_size_view(src_layer, use_mla=True)
                        for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                            assert mem.tensor is not None
                            sl = slot_mapping_on_device[s:e]
                            gathered = src_flat.index_select(0, sl)
                            mem.tensor.copy_(
                                gathered.to(mem.tensor.device),
                                non_blocking=True,
                            )

                        target_fmt = MemoryFormat.KV_MLA_FMT
                        for mem in mem_layer:
                            self._validate_format_transition(mem, target_fmt)
                            mem.metadata.fmt = target_fmt
                    else:
                        src_k_flat, src_v_flat = _get_head_size_view(
                            src_layer, use_mla=False
                        )
                        for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                            assert mem.tensor is not None
                            sl = slot_mapping_on_device[s:e]
                            k = src_k_flat.index_select(0, sl)
                            v = src_v_flat.index_select(0, sl)
                            _copy_kv_into_mem(mem.tensor, k, v)

                if sync:
                    self.store_stream.synchronize()
                yield
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_to_gpu(
        self,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """Wrapper that delegates to the generator implementation."""
        return self._batched_to_gpu_gen(starts=starts, ends=ends, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Return the shape of a single-layer KV buffer."""
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
