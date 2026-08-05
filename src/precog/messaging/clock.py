"""
ShmBeat — shared-memory tick counter with per-level back-pressure barriers.

L0 is the writer, calling beat() each cycle. L1+ are readers, calling
wait_tick() / mark_done() to synchronize with the global tick.

Shared memory layout (16 + 8*num_levels bytes):
  offset  0: int64   tick_count
  offset  8: float64 sim_time
  offset 16: int64   completed[0]  (always == tick_count for L0)
  offset 24: int64   completed[1]  (L1's last completed tick)
  offset 32: int64   completed[2]  (L2's last completed tick)
  ...
"""

from __future__ import annotations

import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Optional


_HEADER_SIZE = 16  # tick_count + sim_time
_SLOT_SIZE = 8     # one int64 per level


def _pack_int64(buf, offset: int, value: int) -> None:
    struct.pack_into("<q", buf, offset, value)


def _unpack_int64(buf, offset: int) -> int:
    return struct.unpack("<q", buf[offset : offset + 8])[0]


def _pack_float64(buf, offset: int, value: float) -> None:
    struct.pack_into("<d", buf, offset, value)


def _unpack_float64(buf, offset: int) -> float:
    return struct.unpack("<d", buf[offset : offset + 8])[0]


class ShmBeat:
    """Shared-memory monotonic tick with back-pressure barriers."""

    def __init__(self, name: str, num_levels: int, *, create: bool = False,
                 base_frequency: float = 0.0):
        self._num_levels = num_levels
        self._total_bytes = _HEADER_SIZE + num_levels * _SLOT_SIZE
        shm_name = f"shm_beat_{name}"

        if create:
            self._shm = SharedMemory(name=shm_name, create=True,
                                     size=self._total_bytes)
            self._buf = self._shm.buf
            _pack_int64(self._buf, 0, 0)           # tick_count = 0
            _pack_float64(self._buf, 8, 0.0)       # sim_time = 0.0
            for i in range(num_levels):
                _pack_int64(self._buf, _HEADER_SIZE + i * _SLOT_SIZE, -1)
            self._base_dt = 1.0 / base_frequency if base_frequency > 0 else 0.0
            self._is_writer = True
        else:
            self._shm = SharedMemory(name=shm_name)
            self._buf = self._shm.buf
            self._base_dt = 1.0 / base_frequency if base_frequency > 0 else 0.0
            self._is_writer = False

        self._closed = False

    # -- properties --------------------------------------------------------

    @property
    def tick_count(self) -> int:
        return _unpack_int64(self._buf, 0)

    @property
    def sim_time(self) -> float:
        return _unpack_float64(self._buf, 8)

    @property
    def base_dt(self) -> float:
        return self._base_dt

    # -- writer (L0) -------------------------------------------------------

    def beat(self) -> int:
        """Advance tick by one. Blocks until all L1+ complete prior tick."""
        tick = self.tick_count

        # Barrier: wait for every non-L0 level to reach or pass this tick
        while True:
            all_done = True
            for i in range(1, self._num_levels):
                off = _HEADER_SIZE + i * _SLOT_SIZE
                if _unpack_int64(self._buf, off) < tick:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(0.0001)

        tick += 1
        _pack_int64(self._buf, 0, tick)
        _pack_float64(self._buf, 8, self.sim_time + self._base_dt)
        _pack_int64(self._buf, _HEADER_SIZE + 0 * _SLOT_SIZE, tick)
        return tick

    # -- reader (L1+) ------------------------------------------------------

    def wait_tick(self, level_idx: int) -> None:
        """Block until a new tick is available for this level."""
        off = _HEADER_SIZE + level_idx * _SLOT_SIZE
        while True:
            tick = self.tick_count
            done = _unpack_int64(self._buf, off)
            if tick > done:
                return
            time.sleep(0.0001)

    def mark_done(self, level_idx: int) -> None:
        """Signal that this level has finished processing the current tick."""
        off = _HEADER_SIZE + level_idx * _SLOT_SIZE
        _pack_int64(self._buf, off, self.tick_count)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (f"ShmBeat(tick={self.tick_count}, "
                f"sim={self.sim_time:.4f}s, "
                f"levels={self._num_levels})")
