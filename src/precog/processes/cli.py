"""
CLI entry point for the pipelined actor architecture.

Replaces the old synchronous main.py with a multiprocessing launcher.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Precog pipelined actor — hierarchical predictive coding with decoupled learning"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the web dashboard",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for learner and upper-level processes (default: cpu)",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=2,
        help="Number of PC levels (default: 2)",
    )
    parser.add_argument(
        "--ctrl-level",
        type=int,
        default=0,
        help="Index of the control level (default: 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--ctrl-weight",
        type=float,
        default=1.0,
        help="Imitation loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--staleness-max",
        type=int,
        default=500,
        help="Max tolerable staleness gap (default: 500)",
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=0.1,
        help="Weight sync interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--level-frequencies",
        type=str,
        default="200,100",
        help="Comma-separated level frequencies in Hz (default: 200,100)",
    )
    parser.add_argument(
        "--rt-priority",
        type=int,
        default=0,
        help="SCHED_FIFO priority for L0 (1-99). 0 = no change (default: 0)",
    )
    parser.add_argument(
        "--rt-core",
        type=int,
        default=None,
        help="CPU core to pin L0 to (default: no pinning)",
    )

    args = parser.parse_args()

    level_frequencies = [float(f.strip()) for f in args.level_frequencies.split(",")]

    if len(level_frequencies) != args.num_levels:
        print(
            f"Warning: level_frequencies count ({len(level_frequencies)}) "
            f"!= num_levels ({args.num_levels}), using num_levels"
        )
        # Auto-generate frequencies
        base = 200.0
        level_frequencies = [base / (2**i) for i in range(args.num_levels)]

    from precog.processes.launcher import Launcher

    launcher = Launcher(
        num_levels=args.num_levels,
        ctrl_level_idx=args.ctrl_level,
        device=args.device,
        headless=args.headless,
        dashboard_port=args.dashboard_port,
        level_frequencies=level_frequencies,
        learning_rate=args.lr,
        ctrl_weight=args.ctrl_weight,
        staleness_max=args.staleness_max,
        sync_interval=args.sync_interval,
        rt_priority=args.rt_priority,
        rt_core=args.rt_core,
    )

    try:
        launcher.start()
    except KeyboardInterrupt:
        print("\nShutting down.")
        launcher.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
