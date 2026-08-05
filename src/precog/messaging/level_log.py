"""
LevelLog — shared-memory per-level diagnostics slot.

Each level process writes diagnostics (idle time, surprise, losses) to its
own shared-memory segment at a configurable throttled rate (default 2 Hz).

The dashboard (or any other reader) attaches to all level log slots and
polls them for display.

Shared memory layout (88 bytes per slot):
  offset  0: int32   level_idx
  offset  4: int32   _pad
  offset  8: int64   timestamp_ns       (monotonic clock, ns)
  offset 16: int64   tick_count
  offset 24: float64 sim_time
  offset 32: float64 idle_time_us       (time spent waiting for other levels)
  offset 40: float64 surprise           (PC prediction error magnitude)
  offset 48: float64 loss_pred          (SSM prediction loss)
  offset 56: float64 loss_sigreg        (SIG regularisation loss)
  offset 64: float64 loss_task          (task objective loss)
  offset 72: float64 loss_translate     (translation / autoencoding loss)
  offset 80: float64 loss_total         (total combined loss)
"""

from __future__ import annotations

import struct
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, List, Optional


_LEVEL_IDX_OFF = 0  # int32
_PAD_OFF = 4  # int32 (padding)
_TIMESTAMP_NS_OFF = 8  # int64
_TICK_COUNT_OFF = 16  # int64
_SIM_TIME_OFF = 24  # float64
_IDLE_TIME_US_OFF = 32  # float64
_SURPRISE_OFF = 40  # float64
_LOSS_PRED_OFF = 48  # float64
_LOSS_SIGREG_OFF = 56  # float64
_LOSS_TASK_OFF = 64  # float64
_LOSS_TRANSLATE_OFF = 72  # float64
_LOSS_TOTAL_OFF = 80  # float64
_TOTAL_BYTES = 88

_DEFAULT_INTERVAL_S = 0.5

_FMT = "<2i3d7d"  # int32, int32, int64, int64, 7×float64 = 4+4+8+8+56 = 80 ... no that's not right

# Use explicit struct layout: 2i (8) + 2q (16) + 7d (56) = 80 ... but I have 88
# Let me just pack manually like ShmBeat does.


def _pack_int32(buf, offset: int, value: int) -> None:
    struct.pack_into("<i", buf, offset, value)


def _unpack_int32(buf, offset: int) -> int:
    return struct.unpack("<i", buf[offset : offset + 4])[0]


def _pack_int64(buf, offset: int, value: int) -> None:
    struct.pack_into("<q", buf, offset, value)


def _unpack_int64(buf, offset: int) -> int:
    return struct.unpack("<q", buf[offset : offset + 8])[0]


def _pack_float64(buf, offset: int, value: float) -> None:
    struct.pack_into("<d", buf, offset, value)


def _unpack_float64(buf, offset: int) -> float:
    return struct.unpack("<d", buf[offset : offset + 8])[0]


def _time_ns() -> int:
    return time.perf_counter_ns()


class LevelLog:
    """Shared-memory per-level diagnostics slot with throttled writes.

    Each level gets its own segment named ``shm_level_log_{level_idx}``.
    The writer calls ``write(...)`` periodically (the method itself throttles
    to ``interval_s``).  Readers call ``read()`` to get a snapshot dict.

    Args:
        level_idx: Index of this level in the hierarchy.
        interval_s: Minimum interval between shared-memory updates (default 0.5).
    """

    def __init__(self, level_idx: int, interval_s: float = _DEFAULT_INTERVAL_S):
        self._level_idx = level_idx
        self._interval_s = interval_s
        self._last_write_ns: int = 0

        shm_name = f"shm_level_log_{level_idx}"
        self._shm = SharedMemory(name=shm_name, create=True, size=_TOTAL_BYTES)
        self._buf = self._shm.buf
        self._closed = False

        _pack_int32(self._buf, _LEVEL_IDX_OFF, level_idx)
        _pack_int32(self._buf, _PAD_OFF, 0)
        _pack_int64(self._buf, _TIMESTAMP_NS_OFF, 0)
        _pack_int64(self._buf, _TICK_COUNT_OFF, -1)
        _pack_float64(self._buf, _SIM_TIME_OFF, 0.0)
        _pack_float64(self._buf, _IDLE_TIME_US_OFF, 0.0)
        _pack_float64(self._buf, _SURPRISE_OFF, 0.0)
        _pack_float64(self._buf, _LOSS_PRED_OFF, 0.0)
        _pack_float64(self._buf, _LOSS_SIGREG_OFF, 0.0)
        _pack_float64(self._buf, _LOSS_TASK_OFF, 0.0)
        _pack_float64(self._buf, _LOSS_TRANSLATE_OFF, 0.0)
        _pack_float64(self._buf, _LOSS_TOTAL_OFF, 0.0)

    # -- factory ------------------------------------------------------------

    @classmethod
    def attach(cls, level_idx: int) -> LevelLog:
        """Attach to an existing LevelLog from another process."""
        log = cls.__new__(cls)
        log._level_idx = level_idx
        log._interval_s = 0.0
        log._last_write_ns = 0
        log._shm = SharedMemory(name=f"shm_level_log_{level_idx}")
        log._buf = log._shm.buf
        log._closed = False
        return log

    # -- write --------------------------------------------------------------

    def write(
        self,
        *,
        tick_count: int,
        sim_time: float,
        idle_time_us: float,
        surprise: float,
        loss_pred: float = 0.0,
        loss_sigreg: float = 0.0,
        loss_task: float = 0.0,
        loss_translate: float = 0.0,
        loss_total: float = 0.0,
    ) -> bool:
        """Write a diagnostics snapshot (throttled to ``interval_s``).

        Returns True if the write was performed, False if it was
        skipped (throttled).
        """
        now = _time_ns()
        if now - self._last_write_ns < self._interval_s * 1_000_000_000:
            return False

        self._last_write_ns = now
        _pack_int64(self._buf, _TIMESTAMP_NS_OFF, now)
        _pack_int64(self._buf, _TICK_COUNT_OFF, tick_count)
        _pack_float64(self._buf, _SIM_TIME_OFF, sim_time)
        _pack_float64(self._buf, _IDLE_TIME_US_OFF, idle_time_us)
        _pack_float64(self._buf, _SURPRISE_OFF, surprise)
        _pack_float64(self._buf, _LOSS_PRED_OFF, loss_pred)
        _pack_float64(self._buf, _LOSS_SIGREG_OFF, loss_sigreg)
        _pack_float64(self._buf, _LOSS_TASK_OFF, loss_task)
        _pack_float64(self._buf, _LOSS_TRANSLATE_OFF, loss_translate)
        _pack_float64(self._buf, _LOSS_TOTAL_OFF, loss_total)
        return True

    # -- read ---------------------------------------------------------------

    def read(self) -> Optional[Dict[str, float]]:
        """Read a diagnostics snapshot.

        Returns a dict of field → value, or None if no data has
        ever been written (tick_count == -1).
        """
        tick_count = _unpack_int64(self._buf, _TICK_COUNT_OFF)
        if tick_count < 0:
            return None

        return {
            "level_idx": float(_unpack_int32(self._buf, _LEVEL_IDX_OFF)),
            "tick_count": float(tick_count),
            "timestamp_ns": float(_unpack_int64(self._buf, _TIMESTAMP_NS_OFF)),
            "sim_time": _unpack_float64(self._buf, _SIM_TIME_OFF),
            "idle_time_us": _unpack_float64(self._buf, _IDLE_TIME_US_OFF),
            "surprise": _unpack_float64(self._buf, _SURPRISE_OFF),
            "loss_pred": _unpack_float64(self._buf, _LOSS_PRED_OFF),
            "loss_sigreg": _unpack_float64(self._buf, _LOSS_SIGREG_OFF),
            "loss_task": _unpack_float64(self._buf, _LOSS_TASK_OFF),
            "loss_translate": _unpack_float64(self._buf, _LOSS_TRANSLATE_OFF),
            "loss_total": _unpack_float64(self._buf, _LOSS_TOTAL_OFF),
        }

    # -- bulk reader helper -------------------------------------------------

    @staticmethod
    def read_all(num_levels: int) -> List[Optional[Dict[str, float]]]:
        """Convenience: attach to all levels 0..num_levels-1 and read each.

        Returns a list parallel to level indices; entries are None for
        levels that have never written.
        """
        results: List[Optional[Dict[str, float]]] = []
        for i in range(num_levels):
            try:
                log = LevelLog.attach(i)
                data = log.read()
                log.close()
                results.append(data)
            except FileNotFoundError:
                results.append(None)
        return results

    # -- lifecycle ----------------------------------------------------------

    @property
    def level_idx(self) -> int:
        return self._level_idx

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._buf = None
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"LevelLog(level={self._level_idx}, interval={self._interval_s}s)"
