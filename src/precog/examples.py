"""
Usage examples for the hierarchical PC world model.
"""

import torch
from precog.model import ClockConfig, HierarchicalClock


# -----------------------------------------------------------------------
# 8. Clock standalone (use without a runner)
# -----------------------------------------------------------------------


def example_clock_standalone():
    """The clock can be used independently for any multi-rate scheduling."""

    cfg = ClockConfig(level_frequencies=[100, 10, 1], time_scale=1.0)
    clock = HierarchicalClock(cfg)

    for _ in range(110):
        clock.tick()
        flags = [clock.should_update(i) for i in range(3)]
        if any(flags):
            print(
                f"tick={clock._tick_count:3d}  "
                f"L0={'✓' if flags[0] else '·'}  "
                f"L1={'✓' if flags[1] else '·'}  "
                f"L2={'✓' if flags[2] else '·'}"
            )
