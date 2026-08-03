"""
Seqlock-protected single-slot shared-memory tensor channel.

Writer-latest-wins pattern for inter-process handoffs:
  - Writer copies data, then bumps a monotonic generation counter.
  - Reader snapshots the counter, copies data, then re-reads the counter.
    If counter changed or was odd (writer active), retry.
  - Reader uses short bounded spin-then-yield: ~50us spin, then os.sched_yield().

The generation counter lives in the shared-memory buffer itself
(last 4 bytes), so attacher and creator share it automatically.

Used for:
  - Upward channels (z_L → L+1)
  - Downward channels (pred_from_above → L-1)
"""

from __future__ import annotations

import os
import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import numpy as np
import torch


_WRITER_RETRY_COUNT = 3
_READER_SPIN_US = 50
_READER_SPIN_ITERATIONS = 10
_READER_RETRY_COUNT = 16
_GEN_OFFSET = 0
_GEN_BYTES = 4


def _time_us() -> float:
    return time.perf_counter() * 1_000_000


def _encode_u32(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)


def _decode_u32(b: memoryview) -> int:
    return struct.unpack("<I", b.cast("B")[:4])[0]


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
        self._dtype = np.dtype(dtype)
        self._payload_bytes = int(np.prod(shape)) * self._dtype.itemsize
        self._total_bytes = self._payload_bytes + _GEN_BYTES

        shm_name = f"shm_slot_{name}"
        self._shm = SharedMemory(name=shm_name, create=True, size=self._total_bytes)
        self._gen_view = self._shm.buf[_GEN_OFFSET:_GEN_OFFSET + _GEN_BYTES]

        self._name = name
        self._shm_name = shm_name
        self._closed = False

    @classmethod
    def attach(cls, name: str, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> "ShmTensorSlot":
        """Attach to an existing ShmTensorSlot from another process."""
        d = np.dtype(dtype)
        payload_bytes = int(np.prod(shape)) * d.itemsize
        total_bytes = payload_bytes + _GEN_BYTES

        slot = cls.__new__(cls)
        slot._shape = shape
        slot._dtype = d
        slot._payload_bytes = payload_bytes
        slot._total_bytes = total_bytes
        slot._shm = SharedMemory(name=f"shm_slot_{name}")
        slot._gen_view = slot._shm.buf[_GEN_OFFSET:_GEN_OFFSET + _GEN_BYTES]
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

    def _bump_gen(self) -> None:
        v = _decode_u32(self._gen_view) + 1
        self._gen_view[:] = _encode_u32(v)

    def _read_gen(self) -> int:
        return _decode_u32(self._gen_view)

    def write(self, tensor: torch.Tensor) -> None:
        """Write a tensor to the slot. Blocks briefly during seqlock window."""
        tensor = tensor.detach().cpu().contiguous()
        arr = np.ndarray(
            self._shape, dtype=self._dtype,
            buffer=self._shm.buf[_GEN_BYTES:_GEN_BYTES + self._payload_bytes]
        )

        for _ in range(_WRITER_RETRY_COUNT):
            self._bump_gen()
            np.copyto(arr, tensor.numpy().reshape(self._shape))
            self._bump_gen()
            return

    def read(self, timeout_us: float = 5000) -> Optional[torch.Tensor]:
        """Read the latest tensor from the slot.

        Args:
            timeout_us: Maximum time in microseconds to wait for valid data.

        Returns:
            Tensor copy from shared memory, or None if timeout.
        """
        arr = np.ndarray(
            self._shape, dtype=self._dtype,
            buffer=self._shm.buf[_GEN_BYTES:_GEN_BYTES + self._payload_bytes]
        )
        deadline = _time_us() + timeout_us

        for _ in range(_READER_RETRY_COUNT):
            for __ in range(_READER_SPIN_ITERATIONS):
                if _time_us() >= deadline:
                    return None

                gen_0 = self._read_gen()
                if gen_0 & 1:
                    time.sleep(0)
                    continue

                result = arr.copy()
                gen_1 = self._read_gen()

                if gen_0 == gen_1 and not (gen_0 & 1):
                    return torch.from_numpy(result).clone()

            os.sched_yield()

        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass
