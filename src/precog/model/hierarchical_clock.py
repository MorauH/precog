"""
LevelClock — per-process clock backed by a shared ShmBeat.

L0 (the fastest level) creates a writer clock: it calls tick() which
advances the shared beat, blocking until all slower levels finish.

L1..Ln create reader clocks: they call wait_tick() to block until a new
tick is available, then should_update() to check divisibility, then
mark_done() to signal completion.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from precog.messaging.clock import ShmBeat


class LevelClock:
    """Per-process multi-rate clock backed by a shared ShmBeat.

    Args:
        beat: Shared beat counter.
        divisor: base_frequency / this_level_frequency (1 for L0).
        level_idx: Index of this level in the hierarchy (for completion slot).
    """

    def __init__(self, beat: ShmBeat, divisor: int, level_idx: int = 0):
        self._beat = beat
        self._divisor = divisor
        self._level_idx = level_idx
        self._last_quotient: int = -1
        self._next_tick_time: float = time.perf_counter()

    # -- factories ---------------------------------------------------------

    @classmethod
    def writer(cls, beat: ShmBeat, level_idx: int = 0) -> LevelClock:
        """Create a writer clock for L0 (divisor always 1)."""
        return cls(beat, divisor=1, level_idx=level_idx)

    @classmethod
    def reader(cls, beat: ShmBeat, divisor: int,
               level_idx: int) -> LevelClock:
        """Create a reader clock for L1+."""
        return cls(beat, divisor=divisor, level_idx=level_idx)

    # -- writer (L0) -------------------------------------------------------

    def tick(self) -> int:
        """Advance the shared beat (writer only). Returns new tick count."""
        return self._beat.beat()

    # -- reader (L1+) ------------------------------------------------------

    def wait_tick(self) -> None:
        """Block until a new tick is available (reader only)."""
        self._beat.wait_tick(self._level_idx)

    def mark_done(self) -> None:
        """Signal completion of current tick (reader only)."""
        self._beat.mark_done(self._level_idx)

    # -- shared ------------------------------------------------------------

    def should_update(self) -> bool:
        """True if this level should fire on the current tick.

        Uses integer division on the shared tick counter so that a level
        with divisor D fires exactly once every D ticks.
        """
        q = self._beat.tick_count // self._divisor
        if q > self._last_quotient:
            self._last_quotient = q
            return True
        return False

    @property
    def tick_count(self) -> int:
        return self._beat.tick_count

    @property
    def sim_time(self) -> float:
        return self._beat.sim_time

    @property
    def base_dt(self) -> float:
        return self._beat.base_dt

    # -- pacing ------------------------------------------------------------

    def sleep_until_next_tick(self, time_scale: float = 1.0) -> None:
        """Block until the next base tick should fire (real-time pacing).

        Uses a rolling target so overruns don't cause catch-up bursts.
        """
        scaled_dt = self._beat.base_dt / time_scale
        now = time.perf_counter()
        target = self._next_tick_time
        if target <= now:
            self._next_tick_time = now + scaled_dt
            return
        time.sleep(target - now)
        self._next_tick_time = target + scaled_dt

    def __repr__(self) -> str:
        return (f"LevelClock(tick={self.tick_count}, "
                f"sim={self.sim_time:.4f}s, "
                f"divisor={self._divisor}, "
                f"level={self._level_idx})")
