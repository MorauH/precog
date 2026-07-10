"""
MultiRateRunner
===============
Orchestrates multi-frequency execution of a HierarchicalPCWorldModel.

Design goals
------------
1. **Model stays unchanged.**  The runner sits *around* the model; no changes
   to PCLevel or HierarchicalPCWorldModel are required.
2. **Single-tick API.**  Callers do:
       result = runner.tick(obs_dict, prev_action)
   The runner decides which levels fire, feeds stale outputs for those that
   don't, and returns a unified result dict.
3. **Clean time control.**  Pass ``time_scale`` to run faster / slower than
   real-time.  Use ``runner.clock.set_time_scale(n)`` to change it live.
4. **Easy episode resets.**  ``runner.reset()`` zeros all LevelStates and
   restarts the clock.

Level communication at different rates
---------------------------------------
*Downward prediction* (from level i+1 to level i):
    Level i+1 produces ``last_z_hat_next`` on its slower cadence.  When
    level i fires on a faster tick, it reads the *held* ``last_z_hat_next``
    from the LevelState above.  No interpolation — zero-order hold is the
    right inductive bias for discrete-time predictive coding.

*Upward representation* (from level i to level i+1):
    Level i+1 expects a sequence input.  We accumulate the ``last_z``
    outputs produced by level i since the last time level i+1 ran and
    pass that batch as a short sequence.  If level i+1 fires every K
    base ticks and level i fires every tick, the sequence length is K.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from .hierarchical_clock import ClockConfig, HierarchicalClock
from .level_state import LevelState

if TYPE_CHECKING:
    from .model import HierarchicalPCWorldModel


@dataclass
class RunnerConfig:
    """Configuration knobs for the MultiRateRunner.

    Parameters
    ----------
    level_frequencies:
        Hz for each level, index 0 = lowest/fastest.  E.g. [100, 10, 1].
    time_scale:
        Simulation speed multiplier (1.0 = real-time, 10.0 = 10× faster).
    batch_size:
        Inference batch size (usually 1 for online robot control).
    device:
        Torch device for new tensors.
    accumulate_for_upper:
        If True (default), collect level-i outputs into a sequence and feed
        them to level i+1 when level i+1 fires.  This preserves temporal
        detail across the hierarchy.
        If False, only the most recent level-i output is used (simpler but
        loses within-window dynamics at higher levels).
    """

    level_frequencies: List[float]
    time_scale: float = 1.0
    batch_size: int = 1
    device: str = "cpu"
    accumulate_for_upper: bool = True


@dataclass
class TickResult:
    """Return value of ``MultiRateRunner.tick()``.

    Attributes
    ----------
    action:
        Control output from the control head, shape (B, action_dim).
    level_states:
        Snapshot of all LevelStates after this tick (read-only).
    updated_levels:
        Which level indices actually ran (fired) this tick.
    sim_time:
        Simulated time in seconds at end of this tick.
    total_surprise:
        Sum of squared prediction errors across all levels that fired.
    """

    action: torch.Tensor
    level_states: List[LevelState]
    updated_levels: List[int]
    sim_time: float
    total_surprise: Optional[torch.Tensor]


class MultiRateRunner:
    """Multi-frequency execution wrapper for HierarchicalPCWorldModel.

    Parameters
    ----------
    model:
        An instantiated HierarchicalPCWorldModel.
    runner_cfg:
        Frequency / device / batch config.

    Example
    -------
    >>> cfg = RunnerConfig(level_frequencies=[100, 10, 1], time_scale=5.0)
    >>> runner = MultiRateRunner(model, cfg)
    >>> for obs, prev_act in environment:
    ...     result = runner.tick(obs, prev_act)
    ...     send_to_robot(result.action)
    ...     runner.clock.sleep_until_next_tick()   # optional real-time pacing
    """

    def __init__(self, model, runner_cfg: RunnerConfig):
        self.model = model
        self.cfg = runner_cfg
        self.device = torch.device(runner_cfg.device)

        # Clock
        clock_cfg = ClockConfig(
            level_frequencies=runner_cfg.level_frequencies,
            time_scale=runner_cfg.time_scale,
        )
        self.clock = HierarchicalClock(clock_cfg)

        # Per-level states
        self._level_states: List[LevelState] = self._init_level_states()

        # Accumulation buffers: for each level, a list of z tensors produced
        # since the parent level last ran.
        self._accum_buffers: List[List[torch.Tensor]] = [
            [] for _ in runner_cfg.level_frequencies
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def tick(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
    ) -> TickResult:
        """Advance one base clock tick and update any due levels.

        Parameters
        ----------
        obs_dict:
            Dict of sensor tensors, each shape (B, sensor_dim).
            (No time dimension — the runner adds it.)
        prev_action:
            Previous robot action, shape (B, action_dim), or None.

        Returns
        -------
        TickResult with the latest action and full level state snapshot.
        """
        self.clock.tick()
        current_tick = self.clock._tick_count
        sim_time = self.clock.sim_time()

        updated_levels: List[int] = []
        total_surprise: Optional[torch.Tensor] = None
        action: Optional[torch.Tensor] = None

        # -------------------------------------------------------------- #
        # Level 0 encoding (always runs — it's at the base frequency)
        # -------------------------------------------------------------- #
        z_level0 = self._encode_level0(obs_dict, prev_action)  # (B, d_level0)

        # -------------------------------------------------------------- #
        # Hierarchical update: bottom-up
        # -------------------------------------------------------------- #
        # z_input_for_level[i] = the sequence to feed level i
        # For level 0, it's z_level0 expanded to (B,1,d).
        # For level i>0, it's the accumulated buffer of level i-1 outputs
        # (or just the last output, depending on cfg.accumulate_for_upper).

        z_input = z_level0.unsqueeze(1)  # (B, 1, d_level0)

        surprise_terms: List[torch.Tensor] = []

        for i, level in enumerate(self.model.levels):
            state = self._level_states[i]

            # Accumulate level-i input for level i+1
            # (store the current z_input last-frame for upper consumption)
            self._accum_buffers[i].append(z_input[:, -1].detach())  # (B, d)

            if not self.clock.should_update(i):
                # Level i is NOT due — reuse cached outputs, no forward pass.
                if state.last_z is not None:
                    z_input = state.last_z.unsqueeze(1)
                continue

            updated_levels.append(i)

            # Build the sequence input for this level
            seq = self._pop_sequence_for_level(i, z_input)  # (B, T, d)

            # Downward prediction from level above — compute from held z
            pred_from_above: Optional[torch.Tensor] = None
            if i + 1 < len(self._level_states):
                above_state = self._level_states[i + 1]
                if above_state.last_z is not None:
                    above_level = self.model.levels[i + 1]
                    with torch.no_grad():
                        pfa = above_level.predict_downward(
                            above_state.last_z
                        )  # (B, d_repr_of_current)
                    T = seq.shape[1]
                    pred_from_above = pfa.unsqueeze(1).expand(-1, T, -1)  # (B, T, d)

            # Forward through this level
            z_seq, _, z_hat_next_seq, _, h_new, h_target_new, epsilon = level.forward(
                seq,
                state.hidden,
                state.hidden_target,
                pred_from_above,
            )

            # Update LevelState
            state.update(
                hidden=h_new[:, -1],
                hidden_target=h_target_new[:, -1],
                z=z_seq[:, -1],
                z_hat_next=z_hat_next_seq[:, -1],
                pred_error=epsilon,
                tick=current_tick,
                sim_time=sim_time,
            )

            surprise_terms.append(epsilon.pow(2).mean())

            # Prepare z_input for the level above
            z_input = z_seq  # (B, T, d_repr_i)

        if surprise_terms:
            total_surprise = torch.stack(surprise_terms).sum()

        # -------------------------------------------------------------- #
        # Control head (always runs, uses held values)
        # -------------------------------------------------------------- #
        ctrl_idx = self.model.control_level_idx
        ctrl_state = self._level_states[ctrl_idx]

        z_ctrl = ctrl_state.last_z.unsqueeze(1)  # (B, 1, d_repr)
        z_hat_ctrl = ctrl_state.last_z_hat_next.unsqueeze(1)  # (B, 1, d_repr)

        action = self.model.control_head(torch.cat([z_ctrl, z_hat_ctrl], dim=-1))[
            :, -1
        ]  # (B, action_dim)

        return TickResult(
            action=action,
            level_states=list(self._level_states),  # snapshot (refs, not copies)
            updated_levels=updated_levels,
            sim_time=sim_time,
            total_surprise=total_surprise,
        )

    def reset(self, batch_size: Optional[int] = None):
        """Reset all state for a new episode."""
        if batch_size is not None:
            self.cfg.batch_size = batch_size
        self.clock.reset()
        self._level_states = self._init_level_states()
        self._accum_buffers = [[] for _ in self.cfg.level_frequencies]

    def detach_states(self):
        """Detach hidden states to prevent backprop through time across steps.

        Call this periodically during training (e.g. every N ticks) to keep
        computation graphs bounded.
        """
        for state in self._level_states:
            state.detach_()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _init_level_states(self) -> List[LevelState]:
        states = []
        for i, level in enumerate(self.model.levels):
            d_state = level.ssm.d_state
            d_repr = level.d_repr
            states.append(
                LevelState.init(
                    level_idx=i,
                    d_state=d_state,
                    d_repr=d_repr,
                    batch_size=self.cfg.batch_size,
                    device=self.device,
                )
            )
        return states

    def _encode_level0(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run modality encoders and concatenate. Returns (B, d_level0)."""
        encoded = []
        for name, encoder in self.model.modality_encoders.items():
            if name == "control":
                if prev_action is not None:
                    x = prev_action
                else:
                    ctrl_cfg = self.model.config.control_head
                    x = torch.zeros(1, ctrl_cfg.output_dim, device=self.device)
                    x = x.expand(self.cfg.batch_size, -1)
            else:
                x = obs_dict[name]
            z0 = encoder(x.unsqueeze(1))  # encoder expects (B, T, d)
            encoded.append(z0[:, -1])  # (B, d_out)
        return torch.cat(encoded, dim=-1)  # (B, d_level0)

    def _pop_sequence_for_level(
        self, level_idx: int, fallback_z: torch.Tensor
    ) -> torch.Tensor:
        """Build the sequence input for level_idx and clear its buffer.

        If accumulation is enabled and the buffer is non-empty, stack the
        buffered tensors into a sequence (B, T, d).  Otherwise, use
        fallback_z as a single-step sequence.
        """
        buf = self._accum_buffers[level_idx]

        if self.cfg.accumulate_for_upper and buf:
            seq = torch.stack(buf, dim=1)  # (B, T, d)
            self._accum_buffers[level_idx] = []
            return seq

        # Fallback: single step
        self._accum_buffers[level_idx] = []
        return fallback_z  # already (B, T, d)
