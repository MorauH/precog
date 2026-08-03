"""
HierarchicalClock
=================
Manages multi-rate update scheduling for the hierarchical PC world model.

Responsibilities
----------------
* Converts real or simulated time into discrete ticks at the base (highest) frequency.
* Tells callers whether a given level should fire on the current tick.
* Supports a time-scale factor so simulations can run faster or slower than
  wall-clock time without changing any model logic.

Frequency contract
------------------
Level frequencies must be integer divisors of the base frequency so that
every level fires exactly on a tick boundary.  Example:

    base_freq  = 100 Hz   (level 0 fires every tick)
    level_freq = [100, 10, 1]
    divisors   = [  1, 10, 100]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class ClockConfig:
    """Static configuration for the multi-rate clock.

    Parameters
    ----------
    level_frequencies:
        Hz for each level, descending (index 0 = fastest / lowest level).
        E.g. [100, 10, 1].
    time_scale:
        Simulation speed multiplier.  1.0 = real-time; 10.0 = 10× faster.
        Only affects `sleep_until_next_tick()` – the tick counter itself
        advances independently of wall-clock time when you call `tick()`.
    """

    level_frequencies: List[float]
    time_scale: float = 1.0

    def __post_init__(self):
        if not self.level_frequencies:
            raise ValueError("level_frequencies must not be empty.")
        base = max(self.level_frequencies)
        for freq in self.level_frequencies:
            ratio = base / freq
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"All frequencies must be integer divisors of the base "
                    f"frequency {base} Hz.  Got {freq} Hz."
                )
        if self.time_scale <= 0:
            raise ValueError("time_scale must be positive.")


@dataclass
class HierarchicalClock:
    """Stateful multi-rate tick scheduler.

    Usage
    -----
    >>> cfg = ClockConfig(level_frequencies=[100, 10, 1])
    >>> clock = HierarchicalClock(cfg)
    >>> for _ in range(1000):
    ...     clock.tick()
    ...     if clock.should_update(0):   # 100 Hz level
    ...         ...
    ...     if clock.should_update(1):   # 10 Hz level
    ...         ...
    ...     if clock.should_update(2):   # 1 Hz level
    ...         ...
    ...     clock.sleep_until_next_tick()   # optional real-time pacing
    """

    config: ClockConfig
    _tick_count: int = field(default=0, init=False, repr=False)
    _wall_start: float = field(
        default_factory=time.perf_counter, init=False, repr=False
    )
    _next_tick_time: float = field(
        default_factory=time.perf_counter, init=False, repr=False
    )
    _sim_time: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Derived helpers (computed once)
    # ------------------------------------------------------------------ #

    @property
    def base_frequency(self) -> float:
        """Tick rate of the fastest (lowest) level."""
        return max(self.config.level_frequencies)

    @property
    def base_dt(self) -> float:
        """Wall-clock seconds between base ticks (unscaled)."""
        return 1.0 / self.base_frequency

    @property
    def num_levels(self) -> int:
        return len(self.config.level_frequencies)

    def _divisor(self, level_idx: int) -> int:
        """How many base ticks pass between updates of `level_idx`."""
        return round(self.base_frequency / self.config.level_frequencies[level_idx])

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def tick(self) -> int:
        """Advance the clock by one base tick.  Returns the new tick count."""
        self._tick_count += 1
        self._sim_time += self.base_dt
        return self._tick_count

    def should_update(self, level_idx: int) -> bool:
        """True if `level_idx` fires on the *current* tick (call after tick())."""
        return self._tick_count % self._divisor(level_idx) == 0

    def ticks_until_next_update(self, level_idx: int) -> int:
        """Base ticks until `level_idx` fires again (0 = fires this tick)."""
        d = self._divisor(level_idx)
        return (d - self._tick_count % d) % d

    def sim_time(self) -> float:
        """Elapsed simulated time in seconds."""
        return self._sim_time

    def wall_time(self) -> float:
        """Elapsed wall-clock time in seconds since construction."""
        return time.perf_counter() - self._wall_start

    def reset(self):
        """Reset tick counter and timers (e.g. start of new episode)."""
        now = time.perf_counter()
        self._tick_count = 0
        self._sim_time = 0.0
        self._wall_start = now
        self._next_tick_time = now

    # ------------------------------------------------------------------ #
    # Real-time pacing
    # ------------------------------------------------------------------ #

    def sleep_until_next_tick(self):
        """Block until the next base tick should fire (real-time mode).

        Uses a rolling target adjusted by ``max(target, now)`` so that
        an overrun never causes a catch-up burst — the very next tick
        re-anchors to wall-clock time.

        Call this at the **end** of your control loop body.  If processing
        took longer than one tick period, the call returns immediately.
        """
        scaled_dt = self.base_dt / self.config.time_scale
        now = time.perf_counter()
        target = self._next_tick_time
        if target <= now:
            self._next_tick_time = now + scaled_dt
            return
        time.sleep(target - now)
        self._next_tick_time = target + scaled_dt

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def level_dt(self, level_idx: int) -> float:
        """Simulated seconds between updates for `level_idx`."""
        return 1.0 / self.config.level_frequencies[level_idx]

    def set_time_scale(self, time_scale: float):
        """Hot-swap the simulation speed (takes effect on next sleep call)."""
        if time_scale <= 0:
            raise ValueError("time_scale must be positive.")
        self.config.time_scale = time_scale

    def __repr__(self) -> str:
        return (
            f"HierarchicalClock(tick={self._tick_count}, "
            f"sim={self._sim_time:.4f}s, "
            f"freqs={self.config.level_frequencies}, "
            f"scale={self.config.time_scale}×)"
        )
