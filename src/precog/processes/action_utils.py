from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def action_to_dict(tensor: torch.Tensor, keys: list[str]) -> dict[str, np.ndarray]:
    values = tensor[0].detach().cpu().numpy()
    if values.ndim == 0:
        values = np.array([values.item()])
    return {k: values[i].item() for i, k in enumerate(keys)}


def action_from_obs(
    obs: dict, action_keys: list[str], device: torch.device
) -> Optional[torch.Tensor]:
    values = []
    for k in action_keys:
        v = obs.get(f"expert_{k}")
        if v is None:
            return None
        val = v.squeeze()
        if val.ndim == 0:
            val = val.unsqueeze(-1)
        values.append(val)
    return torch.stack(values, dim=-1).to(device)
