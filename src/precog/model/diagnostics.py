"""
Diagnostics collector for online model monitoring.

Collects per-tick statistics (representation health, gradients, tick rate,
surprise) and periodically emits formatted reports.

Usage::

    diag = DiagnosticsCollector(model)
    # Inside tick loop:
    t0 = time.monotonic()
    result = runner.tick(obs, prev_action, diagnostics=diag)
    if diag.should_report():
        print(diag.report())
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, TextIO

import torch

if TYPE_CHECKING:
    from .model import HierarchicalPCWorldModel
    from .multi_rate_runner import MultiRateRunner, TickResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticsConfig:
    enabled: bool = True
    detail_level: str = "full"  # "full" | "light" (hz + action + prev only)
    report_interval_seconds: float = 2.0
    window_ticks: int = 600
    stats_interval: int = 10  # compute expensive stats every N ticks
    compute_grad_norms: bool = True
    compute_param_norms: bool = True
    compute_sigreg: bool = True
    log_to_csv: bool = False
    csv_path: str = "diagnostics.csv"


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------


@dataclass
class _LevelAccumulator:
    updates: int = 0  # ticks this level updated in window
    expensive_samples: int = 0  # ticks with grad/param/sigreg data
    surprise_sum: float = 0.0
    surprise_min: float = float("inf")
    surprise_max: float = float("-inf")
    z_norm_sum: float = 0.0
    z_norm_min: float = float("inf")
    z_norm_max: float = float("-inf")
    z_std_sum: float = 0.0
    z_dead_sum: float = 0.0
    h_norm_sum: float = 0.0
    alpha_last: float = 0.0
    grad_norm_sum: float = 0.0
    grad_norm_max: float = -1.0
    param_norm_sum: float = 0.0
    sigreg_avg_std_sum: float = 0.0
    sigreg_min_std_last: float = 0.0
    sigreg_off_rms_sum: float = 0.0
    sigreg_dim_below_sum: float = 0.0


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class DiagnosticsCollector:
    """Lightweight per-tick diagnostics collector and reporter."""

    def __init__(
        self,
        model: HierarchicalPCWorldModel,
        config: DiagnosticsConfig | None = None,
    ):
        self.model = model
        self.cfg = config or DiagnosticsConfig()
        self.num_levels = len(model.levels)

        self._total_ticks = 0
        self._window_ticks = 0
        self._stats_counter = 0  # for periodic expensive stats
        self._proc_time_sum = 0.0
        self._proc_time_max = 0.0
        self._window_start_wall = time.monotonic()
        self._last_report_wall = time.monotonic()

        self._levels: List[_LevelAccumulator] = [
            _LevelAccumulator() for _ in range(self.num_levels)
        ]

        self._csv_fh: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        if self.cfg.log_to_csv:
            self._csv_fh = open(self.cfg.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_fh,
                fieldnames=[
                    "elapsed",
                    "tick",
                    "sim_time",
                    "level",
                    "surprise",
                    "z_norm",
                    "z_std",
                    "z_dead",
                    "h_norm",
                    "alpha",
                    "grad_norm",
                    "param_norm",
                    "sigreg_avg_std",
                    "sigreg_min_std",
                    "sigreg_off_rms",
                    "sigreg_dim_below",
                    "proc_time",
                ],
            )
            self._csv_writer.writeheader()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_tick(
        self,
        result: TickResult,
        runner: MultiRateRunner,
        *,
        process_time: float | None = None,
    ):
        """Call from inside tick() after the forward+backward pass."""
        if not self.cfg.enabled:
            return

        self._total_ticks += 1
        self._window_ticks += 1
        self._stats_counter += 1

        do_expensive = self._stats_counter % self.cfg.stats_interval == 0

        if process_time is not None:
            self._proc_time_sum += process_time
            if process_time > self._proc_time_max:
                self._proc_time_max = process_time

        for i in result.updated_levels:
            level = self.model.levels[i]
            state = result.level_states[i]
            acc = self._levels[i]
            acc.updates += 1

            # -- surprise ----------------------------------------------------
            if state.last_pred_error is not None:
                eps = state.last_pred_error.pow(2).mean().item()
            else:
                eps = 0.0
            acc.surprise_sum += eps
            acc.surprise_min = min(acc.surprise_min, eps)
            acc.surprise_max = max(acc.surprise_max, eps)

            if self.cfg.detail_level == "light":
                continue

            # -- representation norms ----------------------------------------
            z: torch.Tensor = state.last_z  # (B, d_repr)
            if z is None:
                continue
            z_norms = z.norm(dim=-1)  # (B,)
            z_norm = z_norms.mean().item()
            z_std = z.std(dim=-1).mean().item()
            z_dead = (z.abs() < 1e-6).float().mean().item()

            acc.z_norm_sum += z_norm
            acc.z_norm_min = min(acc.z_norm_min, z_norm)
            acc.z_norm_max = max(acc.z_norm_max, z_norm)
            acc.z_std_sum += z_std
            acc.z_dead_sum += z_dead

            # -- hidden state norm -------------------------------------------
            acc.h_norm_sum += state.hidden.norm(dim=-1).mean().item()

            # -- error correction alpha --------------------------------------
            acc.alpha_last = float(torch.tanh(level.alpha).item())

            # -- expensive stats (periodic) ----------------------------------
            if do_expensive:
                acc.expensive_samples += 1

                if self.cfg.compute_grad_norms:
                    g_norm = _total_grad_norm(level)
                    acc.grad_norm_sum += g_norm
                    acc.grad_norm_max = max(acc.grad_norm_max, g_norm)

                if self.cfg.compute_param_norms:
                    acc.param_norm_sum += _total_param_norm(level)

                if self.cfg.compute_sigreg and hasattr(level, "sigreg"):
                    metrics = level.sigreg.covariance_metrics()
                    acc.sigreg_avg_std_sum += metrics["sigreg_avg_std"]
                    acc.sigreg_min_std_last = metrics["sigreg_min_std"]
                    acc.sigreg_off_rms_sum += metrics["sigreg_off_rms"]
                    acc.sigreg_dim_below_sum += metrics["sigreg_dim_below_thresh"]

            # -- CSV ---------------------------------------------------------
            if self._csv_writer is not None:
                sigreg_avg_std = acc.sigreg_avg_std_sum / max(acc.expensive_samples, 1)
                self._csv_writer.writerow(
                    {
                        "elapsed": round(time.monotonic() - self._window_start_wall, 4),
                        "tick": self._total_ticks,
                        "sim_time": round(result.sim_time, 4),
                        "level": i,
                        "surprise": round(eps, 6),
                        "z_norm": round(z_norm, 4),
                        "z_std": round(z_std, 4),
                        "z_dead": round(z_dead, 4),
                        "h_norm": round(acc.h_norm_sum / acc.updates, 4),
                        "alpha": round(acc.alpha_last, 4),
                        "grad_norm": round(acc.grad_norm_sum / acc.updates, 4)
                        if acc.updates > 0
                        else 0.0,
                        "param_norm": round(
                            acc.param_norm_sum / max(acc.expensive_samples, 1), 4
                        ),
                        "sigreg_avg_std": round(sigreg_avg_std, 4),
                        "sigreg_min_std": round(acc.sigreg_min_std_last, 4),
                        "sigreg_off_rms": round(
                            acc.sigreg_off_rms_sum / max(acc.expensive_samples, 1), 4
                        ),
                        "sigreg_dim_below": round(
                            acc.sigreg_dim_below_sum / max(acc.expensive_samples, 1), 4
                        ),
                        "proc_time": round(process_time, 6)
                        if process_time is not None
                        else 0.0,
                    }
                )

        # Trim window if needed
        if self._window_ticks > self.cfg.window_ticks:
            self._reset_window()

    @property
    def tick_rate(self) -> float:
        elapsed = time.monotonic() - self._window_start_wall
        return self._window_ticks / elapsed if elapsed > 0 else 0.0

    def should_report(self) -> bool:
        if not self.cfg.enabled:
            return False
        return (
            time.monotonic() - self._last_report_wall
            >= self.cfg.report_interval_seconds
        )

    def report(self) -> str:
        """Emit a formatted multi-line report and reset accumulators."""
        self._last_report_wall = time.monotonic()
        lines = self._build_report_lines()
        self._reset_window()
        return "\n".join(lines)

    def close(self):
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _reset_window(self):
        """Reset all window accumulators."""
        self._window_ticks = 0
        self._proc_time_sum = 0.0
        self._proc_time_max = 0.0
        self._window_start_wall = time.monotonic()
        for acc in self._levels:
            acc.updates = 0
            acc.expensive_samples = 0
            acc.surprise_sum = 0.0
            acc.surprise_min = float("inf")
            acc.surprise_max = float("-inf")
            acc.z_norm_sum = 0.0
            acc.z_norm_min = float("inf")
            acc.z_norm_max = float("-inf")
            acc.z_std_sum = 0.0
            acc.z_dead_sum = 0.0
            acc.h_norm_sum = 0.0
            acc.grad_norm_sum = 0.0
            acc.grad_norm_max = -1.0
            acc.param_norm_sum = 0.0
            acc.sigreg_avg_std_sum = 0.0
            acc.sigreg_min_std_last = 0.0
            acc.sigreg_off_rms_sum = 0.0
            acc.sigreg_dim_below_sum = 0.0

    def _build_report_lines(self) -> List[str]:
        elapsed = time.monotonic() - self._window_start_wall
        tick_rate = self._window_ticks / elapsed if elapsed > 0 else 0.0
        avg_proc = (
            self._proc_time_sum / self._window_ticks if self._window_ticks > 0 else 0.0
        )

        lines = [
            "",
            f"── diagnostics  (t={self._total_ticks}, window={self._window_ticks} ticks in {elapsed:.1f}s) ──",
            f"  tick-rate:  {tick_rate:6.1f} Hz  |  proc-time:  avg={avg_proc * 1000:.2f}ms  max={self._proc_time_max * 1000:.2f}ms",
        ]

        if tick_rate > 0:
            min_interval = 1.0 / tick_rate
            overshoot = max(0, avg_proc - min_interval) * 1000
            lines.append(
                f"  interval:   {min_interval * 1000:.2f}ms nominal  |  overshoot:  {overshoot:+.2f}ms"
            )
            # For multi-rate: show per-level effective update rate
            for i, acc in enumerate(self._levels):
                eff_hz = acc.updates / elapsed if elapsed > 0 else 0.0
                lines.append(
                    f"  level[{i}] effective:  {eff_hz:6.1f} Hz  "
                    f"({acc.updates} updates)"
                )

        lines.append("")
        lines.append(
            f"  {'level':<7s} {'surprise':>9s} {'|z|':>9s} {'std(z)':>9s} "
            f"{'|h|':>9s} {'dead%':>6s} {'α':>7s} {'|grad|':>9s} {'|param|':>9s}"
        )
        lines.append("  " + "-" * 85)

        for i, acc in enumerate(self._levels):
            if acc.updates == 0:
                continue
            n = acc.updates
            surprise = f"{acc.surprise_sum / n:.4f}"
            z_norm = f"{acc.z_norm_sum / n:.2f}"
            z_std = f"{acc.z_std_sum / n:.3f}"
            h_norm = f"{acc.h_norm_sum / n:.2f}"
            dead = f"{acc.z_dead_sum / n * 100:.1f}%"
            alpha = f"{acc.alpha_last:+.3f}"
            m = max(acc.expensive_samples, 1)
            grad = (
                f"{acc.grad_norm_sum / m:.2f}"
                if self.cfg.compute_grad_norms
                else "    —"
            )
            param = (
                f"{acc.param_norm_sum / m:.2f}"
                if self.cfg.compute_param_norms
                else "    —"
            )

            lines.append(
                f"  L{i:<6d} {surprise:>9s} {z_norm:>9s} {z_std:>9s} "
                f"{h_norm:>9s} {dead:>6s} {alpha:>7s} {grad:>9s} {param:>9s}"
            )

            # Collapse warnings
            warnings = []
            z_avg = acc.z_norm_sum / n
            if z_avg < 1e-3:
                warnings.append(f"REPR COLLAPSE (|z|={z_avg:.1e})")
            z_std_avg = acc.z_std_sum / n
            if z_std_avg < 1e-3:
                warnings.append(f"DIM COLLAPSE (σ(z)={z_std_avg:.1e})")
            if (
                n > 1
                and acc.surprise_min > 0
                and acc.surprise_max - acc.surprise_min < 1e-6
            ):
                warnings.append("CONSTANT SURPRISE")
            if acc.grad_norm_max < 1e-6:
                warnings.append(f"VANISHING GRADS")
            if warnings:
                for w in warnings:
                    lines.append(f"         ⚠  {w}")

        if self.cfg.compute_sigreg:
            lines.append("")
            lines.append(
                f"  {'level':<7s} {'SIGReg avgσ':>10s} {'minσ':>10s} {'off-rms':>10s} {'dim<thr%':>8s}"
            )
            lines.append("  " + "-" * 42)
            for i, acc in enumerate(self._levels):
                if acc.updates == 0:
                    continue
                m = max(acc.expensive_samples, 1)
                avg_std = acc.sigreg_avg_std_sum / m
                min_std = acc.sigreg_min_std_last
                off_rms = acc.sigreg_off_rms_sum / m
                dim_below = acc.sigreg_dim_below_sum / m * 100
                lines.append(
                    f"  L{i:<6d} {avg_std:>10.4f} {min_std:>10.4f} {off_rms:>10.4f} {dim_below:>7.1f}%"
                )

        lines.append("")
        return lines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _total_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float(p.grad.norm().item() ** 2)
    return total**0.5


def _total_param_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        total += float(p.data.norm().item() ** 2)
    return total**0.5
