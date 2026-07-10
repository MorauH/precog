import sys
import select
import termios
import tty
import threading
import queue
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
import torch
import yaml

from precog.envs import ROSEnvironment
from precog.model import DEFAULT_CONFIG, HierarchicalPCWorldModel
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims
from precog.model.diagnostics import DiagnosticsCollector, DiagnosticsConfig


ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


class OperationMode(Enum):
    DRIVE = auto()
    IMITATE = auto()


class KeyboardReader:
    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _read_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    ch = sys.stdin.read(1)
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    self._queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)

    def get(self) -> Optional[str]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


def _action_to_dict(tensor: torch.Tensor, keys: list[str]) -> dict[str, np.ndarray]:
    values = tensor[0].cpu().numpy()
    return {k: values[i] for i, k in enumerate(keys)}


def _action_from_obs(
    obs: dict, action_keys: list[str], device: torch.device
) -> Optional[torch.Tensor]:
    values = []
    for k in action_keys:
        v = obs.get(f"expert_{k}")
        if v is None:
            return None
        values.append(v.squeeze())
    return torch.stack(values, dim=-1).unsqueeze(0).to(device)


def main():
    rclpy.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(ENV_CONFIG_PATH) as f:
        env_cfg = yaml.safe_load(f)
    env_shapes = _env_shapes_from_yaml(env_cfg)
    config = resolve_config_dims(DEFAULT_CONFIG, env_shapes)

    print(f"Observation keys: {config.observation_keys}")
    print(f"    d_input = {config.d_input}  (control: {config.control_dim})")
    print(
        f"    Levels: {len(config.level_configs)}, "
        f"control from level {config.control_level_idx}"
    )
    print(f"    imitation_loss_weight: {config.imitation_loss_weight}")

    env = ROSEnvironment(config_path=ENV_CONFIG_PATH, device=device)
    action_keys = [s.key for s in env.action_specs]
    model = HierarchicalPCWorldModel(config).to(device)

    diag_cfg = DiagnosticsConfig(
        enabled=True,
        report_interval_seconds=2.0,
        compute_grad_norms=True,
        compute_param_norms=True,
        compute_ema_alignment=True,
    )
    diagnostics = DiagnosticsCollector(model, diag_cfg)

    runner = model.build_runner(
        level_frequencies=[1000, 100],
        time_scale=1.0,
        batch_size=1,
        device=device,
        online_learning=True,
    )

    kb = KeyboardReader()
    kb.start()

    obs = env.reset()
    prev_action: Optional[torch.Tensor] = None
    mode = OperationMode.IMITATE

    print(f"\n  [m] toggle mode  |  Ctrl+C to quit")
    print(f"  Starting in {mode.name} mode\n")

    tick_count = 0

    try:
        while rclpy.ok():
            ch = kb.get()
            if ch == "m":
                mode = (
                    OperationMode.IMITATE
                    if mode == OperationMode.DRIVE
                    else OperationMode.DRIVE
                )
                print(f"\n  >>> switched to {mode.name} mode\n", flush=True)

            if mode == OperationMode.IMITATE:
                prev_action = _action_from_obs(obs, action_keys, device)

            result = runner.tick(obs, prev_action, diagnostics=diagnostics)

            if mode == OperationMode.IMITATE:
                obs = env.step({})
            else:
                act_dict = _action_to_dict(result.action, action_keys)
                obs = env.step(act_dict)
                prev_action = result.action

            tick_count += 1

            if diagnostics.should_report():
                ma = result.action[0].tolist()
                pa = (
                    prev_action[0].tolist()
                    if prev_action is not None
                    else [float("nan"), float("nan")]
                )
                print(
                    f"\n  tick={tick_count}  mode={mode.name}  "
                    f"model=[{ma[0]:.4f}, {ma[1]:.4f}]  "
                    f"prev=[{pa[0]:.4f}, {pa[1]:.4f}]"
                )
                print(diagnostics.report(), flush=True)

            runner.clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        diagnostics.close()
        kb.stop()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
