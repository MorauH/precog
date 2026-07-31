"""
Multi-slot ring buffer for experience handoff from level processes to learner.

Single-writer (level process), single-reader (learner). No lock needed.
Head/tail are atomic multiprocessing.Values.

Each slot contains per-tick experience data needed for replay by the learner:
  - obs parts, z, epsilon, x_star, x_actual, task_target, h_init, prev_action
  - SIGReg EMA buffers (_mean, _outer)
  - weight version tag (for staleness monitoring)
  - sequence number (for staleness monitoring)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import multiprocessing
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory


@dataclass
class ExperienceSlot:
    """Per-tick data for learner replay."""
    level_idx: int

    obs_parts: Dict[str, torch.Tensor]
    z: torch.Tensor
    epsilon: Optional[torch.Tensor]
    x_star: Optional[torch.Tensor]
    x_actual: Optional[torch.Tensor]
    task_target: Optional[torch.Tensor]
    h_init: torch.Tensor
    prev_action: Optional[torch.Tensor]

    sigreg_mean: Optional[torch.Tensor]
    sigreg_outer: Optional[torch.Tensor]

    weight_version: int
    seq_num: int


class ShmRingBuffer:
    """Fixed-size ring buffer for per-tick experience data.

    Each slot stores a pickled ExperienceSlot in shared memory.
    The learner drains all unread slots in batch for efficient replay.

    Args:
        name: Unique buffer name (used in shm naming).
        capacity: Max number of slots in the ring.
        level_idx: Which level this buffer corresponds to.
    """

    def __init__(self, name: str, capacity: int, level_idx: int):
        self._capacity = capacity
        self._level_idx = level_idx
        self._name = name

        self._head = multiprocessing.Value("I", 0, lock=False)
        self._tail = multiprocessing.Value("I", 0, lock=False)
        self._seq_gen = multiprocessing.Value("Q", 0, lock=False)

        self._slots: list[Optional[ExperienceSlot]] = [None] * capacity

        self._slot_count = multiprocessing.Value("I", 0, lock=False)

    @classmethod
    def attach(cls, name: str, capacity: int, level_idx: int) -> "ShmRingBuffer":
        """Attach to an existing ShmRingBuffer from another process."""
        buf = cls.__new__(cls)
        buf._capacity = capacity
        buf._level_idx = level_idx
        buf._name = name
        buf._head = multiprocessing.Value("I", 0, lock=False)
        buf._tail = multiprocessing.Value("I", 0, lock=False)
        buf._seq_gen = multiprocessing.Value("Q", 0, lock=False)
        buf._slots = [None] * capacity
        buf._slot_count = multiprocessing.Value("I", 0, lock=False)
        return buf

    @property
    def name(self) -> str:
        return self._name

    @property
    def level_idx(self) -> int:
        return self._level_idx

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, experience: ExperienceSlot) -> None:
        """Write one experience tick to the ring buffer. Non-blocking."""
        head = self._head.value
        self._slots[head] = experience

        with self._seq_gen.get_lock():
            self._seq_gen.value += 1
            seq = self._seq_gen.value

        self._slots[head].seq_num = seq

        self._head.value = (head + 1) % self._capacity

        count = self._slot_count.value
        if count < self._capacity:
            self._slot_count.value = count + 1

    def drain(self) -> list[ExperienceSlot]:
        """Read all unread slots. Returns empty list if nothing new."""
        tail = self._tail.value
        head = self._head.value

        if tail == head:
            return []

        results: list[ExperienceSlot] = []
        while tail != head:
            slot = self._slots[tail]
            if slot is not None:
                results.append(slot)
            tail = (tail + 1) % self._capacity

        self._tail.value = head
        count = max(0, self._slot_count.value - len(results))
        self._slot_count.value = count

        return results

    def available(self) -> int:
        """Number of unread slots available."""
        tail = self._tail.value
        head = self._head.value
        if head >= tail:
            return head - tail
        return self._capacity - tail + head

    def close(self) -> None:
        self._slots.clear()


def pack_experience(
    level_idx: int,
    obs_parts: Dict[str, torch.Tensor],
    z: torch.Tensor,
    epsilon: Optional[torch.Tensor],
    x_star: Optional[torch.Tensor],
    x_actual: Optional[torch.Tensor],
    task_target: Optional[torch.Tensor],
    h_init: torch.Tensor,
    prev_action: Optional[torch.Tensor],
    sigreg_mean: Optional[torch.Tensor],
    sigreg_outer: Optional[torch.Tensor],
    weight_version: int,
) -> ExperienceSlot:
    """Pack experience data into an ExperienceSlot. Detaches all tensors."""
    return ExperienceSlot(
        level_idx=level_idx,
        obs_parts={k: v.detach().cpu() for k, v in obs_parts.items()},
        z=z.detach().cpu(),
        epsilon=epsilon.detach().cpu() if epsilon is not None else None,
        x_star=x_star.detach().cpu() if x_star is not None else None,
        x_actual=x_actual.detach().cpu() if x_actual is not None else None,
        task_target=task_target.detach().cpu() if task_target is not None else None,
        h_init=h_init.detach().cpu(),
        prev_action=prev_action.detach().cpu() if prev_action is not None else None,
        sigreg_mean=sigreg_mean.detach().cpu().clone() if sigreg_mean is not None else None,
        sigreg_outer=sigreg_outer.detach().cpu().clone() if sigreg_outer is not None else None,
        weight_version=weight_version,
        seq_num=0,
    )
