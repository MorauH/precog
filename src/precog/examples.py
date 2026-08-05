"""
Usage examples for the hierarchical PC world model.
"""

import torch
from precog.messaging.clock import ShmBeat
from precog.model import LevelClock


# -----------------------------------------------------------------------
# Clock: per-process multi-rate scheduling with shared ShmBeat
# -----------------------------------------------------------------------


def example_clock_standalone():
    """Demonstrate shared clock with L0 (writer) and L1/L2 (readers).

    In a real deployment L0 runs in one process, L1/L2 in separate
    processes, all attached to the same ShmBeat by name.
    """

    beat = ShmBeat("demo", num_levels=3, create=True, base_frequency=100.0)

    l0 = LevelClock.writer(beat, level_idx=0)
    l1 = LevelClock.reader(beat, divisor=10, level_idx=1)
    l2 = LevelClock.reader(beat, divisor=100, level_idx=2)

    for _ in range(110):
        l0.tick()
        flags = [l0.should_update(), l1.should_update(), l2.should_update()]
        if any(flags):
            print(
                f"tick={l0.tick_count:3d}  "
                f"L0={'✓' if flags[0] else '·'}  "
                f"L1={'✓' if flags[1] else '·'}  "
                f"L2={'✓' if flags[2] else '·'}"
            )

    beat.unlink()
