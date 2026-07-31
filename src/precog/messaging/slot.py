"""
Seqlock-protected single-slot shared-memory tensor channel.

Writer-latest-wins pattern for inter-process handoffs:
  - Writer copies data, then bumps a monotonic generation counter.
  - Reader snapshots the counter, copies data, then re-reads the counter.
    If counter changed or was odd (writer active), retry.
  - Reader uses short bounded spin-then-yield: ~50us spin, then os.sched_yield().

Used for:
  - Upward channels (z_L → L+1)
  - Downward channels (pred_from_above → L-1)
"""

from __future__ import annotations

import multiprocessing
import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
import torch


_WRITER_RETRY_COUNT = 3
_READER_SPIN_US = 50
_READER_SPIN_ITERATIONS = 10
_READER_RETRY_COUNT = 16

def _time_us() -> float:
    return time.perf_counter() * 1_000_000


def _atomic_cas_increment(value: multiprocessing.Value) -> int:
    """Busy-looping CAS increment on a multiprocessing.Value with lock=False."""
    with value.get_lock():
        value.value += 2
        return value.value


class ShmTensorSlot:
    """Single-slot, latest-wins tensor channel backed by SharedMemory.

    Uses a seqlock-like protocol:
      - gen is even = valid data, odd = writer active
      - Writer increments gen (odd), copies, increments again (even).
      - Reader reads gen_0, copies data, reads gen_1. If gen_0 != gen_1
        or gen_0 is odd, retry.

    Tensors are always read as torch.Tensor (CPU). For CUDA tensors,
    the writer serializes to CPU before writing.
    """

    def __init__(self, name: str, shape: tuple[int, ...], dtype: np.dtype = np.float32):
        self._shape = shape
        self._dtype = dtype
        self._nbytes = int(np.prod(shape)) * dtype.itemsize

        shm_name = f"shm_slot_{name}"
        self._shm = SharedMemory(name=shm_name, create=True, size=self._nbytes)
        self._gen = multiprocessing.Value("I", 0, lock=True)

        self._name = name
        self._shm_name = shm_name
        self._closed = False

    @classmethod
    def attach(cls, name: str, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> "ShmTensorSlot":
        """Attach to an existing ShmTensorSlot from another process."""
        slot = cls.__new__(cls)
        slot._shape = shape
        slot._dtype = dtype
        slot._nbytes = int(np.prod(shape)) * dtype.itemsize
        slot._shm = SharedMemory(name=f"shm_slot_{name}")
        slot._gen = multiprocessing.Value("I", 0, lock=True)
        slot._name = name
        slot._shm_name = f"shm_slot_{name}"
        slot._closed = False
        return slot

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def write(self, tensor: torch.Tensor) -> None:
        """Write a tensor to the slot. Blocks briefly during seqlock window."""
        tensor = tensor.detach().cpu().contiguous()
        arr = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)

        for _ in range(_WRITER_RETRY_COUNT):
            gen = _atomic_cas_increment(self._gen)
            np.copyto(arr, tensor.numpy().reshape(self._shape))
            _atomic_cas_increment(self._gen)
            return

    def read(self, timeout_us: float = 5000) -> Optional[torch.Tensor]:
        """Read the latest tensor from the slot.

        Args:
            timeout_us: Maximum time in microseconds to wait for valid data.

        Returns:
            Tensor view into shared memory, or None if timeout.
        """
        arr = np.ndarray(self._shape, dtype=self._dtype, buffer=self._shm.buf)
        deadline = _time_us() + timeout_us

        for _ in range(_READER_RETRY_COUNT):
            for __ in range(_READER_SPIN_ITERATIONS):
                if _time_us() >= deadline:
                    return None

                gen_0 = self._gen.value
                if gen_0 & 1:
                    time.sleep(0)
                    continue

                result = arr.copy()
                gen_1 = self._gen.value

                if gen_0 == gen_1 and not (gen_0 & 1):
                    return torch.from_numpy(result).clone()

            os.sched_yield()

        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shm.close()
        self._shm.unlink()

    pass
