# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.ops_interface import TransferDirection
from lmcache.v1.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.memory_management import MemoryObj

if torch.cuda.is_available():
    # First Party
    from lmcache.v1.gpu_connector.cuda_ops import CUDAKernelOps

    _cuda_ops = CUDAKernelOps()


# Helper functions
def lmcache_memcpy_async_h2d(
    memory_obj: MemoryObj,
    gpu_buffer: torch.Tensor,
):
    """Helper function to copy memory object allocated by different
    allocators to GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param MemoryObj memory_obj: The memory object to be copied.
    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data to.
    """
    assert memory_obj.tensor is not None
    assert memory_obj.tensor.numel() == gpu_buffer.numel()
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        _cuda_ops.lmcache_memcpy_async(
            gpu_buffer.data_ptr(),
            memory_obj.tensor.data_ptr(),
            memory_obj.get_size(),
            TransferDirection.H2D,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        gpu_buffer.copy_(memory_obj.tensor, non_blocking=True)


def lmcache_memcpy_async_d2h(
    gpu_buffer: torch.Tensor,
    memory_obj: MemoryObj,
):
    """Helper function to copy memory object allocated by different
    allocators from GPU buffer.

    This function is non-blocking and won't do stream synchronization.

    :param torch.Tensor gpu_buffer: The GPU buffer to copy the data from.
    :param MemoryObj memory_obj: The memory object to be copied to.
    """
    assert memory_obj.tensor is not None
    assert memory_obj.tensor.numel() == gpu_buffer.numel()
    if isinstance(memory_obj.parent(), LazyMemoryAllocator):
        _cuda_ops.lmcache_memcpy_async(
            memory_obj.tensor.data_ptr(),
            gpu_buffer.data_ptr(),
            memory_obj.get_size(),
            TransferDirection.D2H,
            memory_obj.meta.address,
            LazyMemoryAllocator.PIN_CHUNK_SIZE,
        )
    else:
        memory_obj.tensor.copy_(gpu_buffer, non_blocking=True)
