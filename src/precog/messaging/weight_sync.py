"""
Weight synchronisation between Learner and level processes.

Double-buffered state_dict handoff:
  - Learner writes to buffer A, flips active pointer.
  - Level process reads from the buffer the pointer doesn't point to.
  - No lock, no wait on read path.

At current model sizes (tens of KB), full state_dict serialization is used.
Flagged for fp16/delta compression when params reach hundreds of MB.
"""

from __future__ import annotations

import io
import multiprocessing
from typing import Any, Dict, Optional

import torch


def serialize_state_dict(state_dict: Dict[str, Any]) -> bytes:
    """Serialize a PyTorch state_dict to bytes (CPU tensors).

    Handles both cpu and cuda tensors by moving to cpu first.
    """
    cpu_state = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            cpu_state[key] = tensor.detach().cpu().contiguous()
        else:
            cpu_state[key] = tensor

    buf = io.BytesIO()
    torch.save(cpu_state, buf)
    return buf.getvalue()


def deserialize_to_model(
    model_layer: torch.nn.Module,
    data: bytes,
    device: Optional[torch.device] = None,
) -> None:
    """Deserialize weight bytes into a model layer.

    Args:
        model_layer: The nn.Module to load weights into.
        data: Serialized state_dict bytes.
        device: Target device. If None, tensors stay on CPU.
    """
    buf = io.BytesIO(data)
    state_dict = torch.load(buf, weights_only=True)

    if device is not None:
        state_dict = {k: v.to(device) for k, v in state_dict.items()}

    model_layer.load_state_dict(state_dict, strict=False)


def clone_params_to_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract a CPU copy of module parameters as a state_dict-like dict."""
    state = {}
    for name, param in module.named_parameters():
        state[name] = param.data.detach().cpu().clone()
    for name, buf in module.named_buffers():
        state[name] = buf.data.detach().cpu().clone()
    return state


class WeightSync:
    """Double-buffered weight synchronisation for one model level.

    Writes from Learner, reads from level process. No locks on read path.

    Args:
        name: Unique name for this sync channel.
        level_idx: Index of the level this sync channel serves.
    """

    def __init__(self, name: str, level_idx: int):
        self._name = name
        self._level_idx = level_idx

        self._buffer_a = multiprocessing.Array("B", 1024 * 1024, lock=False)
        self._buffer_b = multiprocessing.Array("B", 1024 * 1024, lock=False)

        self._active_idx = multiprocessing.Value("I", 0, lock=False)
        self._version_a = multiprocessing.Value("Q", 0, lock=False)
        self._version_b = multiprocessing.Value("Q", 0, lock=False)

        self._data_len_a = multiprocessing.Value("I", 0, lock=False)
        self._data_len_b = multiprocessing.Value("I", 0, lock=False)

    @classmethod
    def attach(cls, name: str, level_idx: int) -> "WeightSync":
        """Attach to an existing WeightSync channel from another process."""
        ws = cls.__new__(cls)
        ws._name = name
        ws._level_idx = level_idx
        ws._buffer_a = multiprocessing.Array("B", 1024 * 1024, lock=False)
        ws._buffer_b = multiprocessing.Array("B", 1024 * 1024, lock=False)
        ws._active_idx = multiprocessing.Value("I", 0, lock=False)
        ws._version_a = multiprocessing.Value("Q", 0, lock=False)
        ws._version_b = multiprocessing.Value("Q", 0, lock=False)
        ws._data_len_a = multiprocessing.Value("I", 0, lock=False)
        ws._data_len_b = multiprocessing.Value("I", 0, lock=False)
        return ws

    @property
    def name(self) -> str:
        return self._name

    @property
    def level_idx(self) -> int:
        return self._level_idx

    def write_state_dict(self, state_dict: Dict[str, Any]) -> int:
        """Write new weights. Returns the new version tag.

        Writes to the inactive buffer, then flips the active pointer.
        """
        data = serialize_state_dict(state_dict)
        data_len = len(data)

        active = self._active_idx.value
        inactive = 1 - active

        if inactive == 0:
            buf = self._buffer_a
            version = self._version_a
            data_len_field = self._data_len_a
        else:
            buf = self._buffer_b
            version = self._version_b
            data_len_field = self._data_len_b

        nbytes = min(data_len, len(buf))
        buf[:nbytes] = data[:nbytes]
        data_len_field.value = data_len

        with version.get_lock():
            version.value += 1
            new_version = version.value

        self._active_idx.value = inactive

        return new_version

    def read_latest(
        self, model_layer: torch.nn.Module, device: Optional[torch.device] = None
    ) -> bool:
        """Read latest weights into model_layer. Returns True if new weights loaded."""
        active = self._active_idx.value

        if active == 0:
            buf = self._buffer_a
            version = self._version_a
            data_len = self._data_len_a.value
        else:
            buf = self._buffer_b
            version = self._version_b
            data_len = self._data_len_b.value

        nbytes = min(data_len, len(buf))
        if nbytes == 0:
            return False

        data = bytes(buf[:nbytes])
        deserialize_to_model(model_layer, data, device)
        return True

    def read_latest_bytes(self) -> Optional[bytes]:
        """Read latest weight bytes. Returns None if no data yet."""
        active = self._active_idx.value

        if active == 0:
            buf = self._buffer_a
            data_len = self._data_len_a.value
        else:
            buf = self._buffer_b
            data_len = self._data_len_b.value

        nbytes = min(data_len, len(buf))
        if nbytes == 0:
            return None

        return bytes(buf[:nbytes])
