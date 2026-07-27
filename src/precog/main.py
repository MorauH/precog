import sys
import select
import termios
import tty
import argparse
import threading
import queue
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
import torch
import yaml

from precog.envs import ROSEnvironment
from precog.envs.ros.source_selector import SourceSelector
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
    values = tensor[0].detach().cpu().numpy()
    if values.ndim == 0:
        values = np.array([values.item()])
    return {k: values[i].item() for i, k in enumerate(keys)}


def _action_from_obs(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable the web dashboard on port 8080",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8080,
        help="Dashboard port (default: 8080)",
    )
    args = parser.parse_args()

    rclpy.init()

    device = "cpu"
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
    print(f"Action keys: {action_keys}")

    source_selector = SourceSelector()

    model = HierarchicalPCWorldModel(config).to(device)
    model.compile(mode="reduce-overhead")

    diag_cfg = DiagnosticsConfig(
        enabled=True,
        detail_level="light",
        report_interval_seconds=2.0,
        compute_grad_norms=True,
        compute_param_norms=True,
        compute_sigreg=True,
    )
    diagnostics = DiagnosticsCollector(model, diag_cfg)

    runner = model.build_runner(
        level_frequencies=[200, 100],
        time_scale=1.0,
        batch_size=1,
        device=device,
        online_learning=True,
    )

    dashboard = None
    kb: Optional[KeyboardReader] = None
    if args.dashboard:
        from precog.dashboard import DashboardServer

        dashboard = DashboardServer(port=args.dashboard_port)
        dashboard.start()
        print(f"  Dashboard: http://localhost:{args.dashboard_port}")
    else:
        kb = KeyboardReader()
        kb.start()

    obs = env.reset()
    prev_action: Optional[torch.Tensor] = None
    mode = OperationMode.IMITATE

    print(f"\n  Starting in {mode.name} mode")
    if dashboard:
        print(f"  Use the dashboard to control blend ratio and mode")
    else:
        print(f"  [m] toggle mode  |  Ctrl+C to quit")
    print()

    tick_count = 0
    blend_steer = 0.0
    blend_acc = 0.0

    try:
        while rclpy.ok():
            if kb is not None:
                ch = kb.get()
                if ch == "m":
                    if mode == OperationMode.IMITATE:
                        mode = OperationMode.DRIVE
                    else:
                        mode = OperationMode.IMITATE
                    print(f"\n  >>> switched to {mode.name} mode\n", flush=True)

            if dashboard is not None:
                ctrl = dashboard.controls
                new_steer = ctrl.blend_steer_safe
                new_acc = ctrl.blend_acc_safe
                if abs(new_steer - blend_steer) > 1e-6:
                    blend_steer = new_steer
                    source_selector.blend_ratio_steer = blend_steer
                if abs(new_acc - blend_acc) > 1e-6:
                    blend_acc = new_acc
                    source_selector.blend_ratio_acc = blend_acc

                dash_mode = ctrl.mode_safe
                if dash_mode == "IMITATE" and mode == OperationMode.DRIVE:
                    mode = OperationMode.IMITATE
                    print(f"\n  >>> dashboard: switched to IMITATE mode\n", flush=True)
                elif dash_mode == "DRIVE" and mode == OperationMode.IMITATE:
                    mode = OperationMode.DRIVE
                    print(f"\n  >>> dashboard: switched to DRIVE mode\n", flush=True)
                    ctrl.set_mode("DRIVE")

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

            if dashboard is not None and tick_count % 10 == 0:
                ma = result.action[0].tolist()
                pa = (
                    prev_action[0].tolist()
                    if prev_action is not None
                    else [float("nan")] * len(action_keys)
                )
                dashboard.update(
                    {
                        "tick": tick_count,
                        "tick_rate": diagnostics.tick_rate,
                        "mode": mode.name,
                        "blend_ratio_steer": blend_steer,
                        "blend_ratio_acc": blend_acc,
                        "action": ma,
                        "prev_action": pa,
                        "source_selector": source_selector.snapshot(),
                        "surprise": [
                            (
                                diagnostics._levels[i].surprise_sum
                                / max(diagnostics._levels[i].updates, 1)
                                if diagnostics._levels[i].updates > 0
                                else 0.0
                            )
                            for i in range(min(2, diagnostics.num_levels))
                        ],
                    }
                )

            if diagnostics.cfg.detail_level == "light":
                if tick_count % 200 == 0:
                    hz = diagnostics.tick_rate
                    ma = result.action[0].tolist()
                    pa = (
                        prev_action[0].tolist()
                        if prev_action is not None
                        else [float("nan")] * len(action_keys)
                    )
                    model_str = ", ".join(f"{v:.4f}" for v in ma)
                    prev_str = ", ".join(f"{v:.4f}" for v in pa)
                    print(
                        f"\n  tick={tick_count}  hz={hz:5.1f}  mode={mode.name}  "
                        f"blend/s={blend_steer:.2f} blend/a={blend_acc:.2f}  "
                        f"model=[{model_str}]  prev=[{prev_str}]"
                    )
            elif diagnostics.should_report():
                ma = result.action[0].tolist()
                pa = (
                    prev_action[0].tolist()
                    if prev_action is not None
                    else [float("nan")] * len(action_keys)
                )
                model_str = ", ".join(f"{v:.4f}" for v in ma)
                prev_str = ", ".join(f"{v:.4f}" for v in pa)
                print(
                    f"\n  tick={tick_count}  mode={mode.name}  "
                    f"model=[{model_str}]  prev=[{prev_str}]"
                )
                print(diagnostics.report(), flush=True)

            runner.clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        diagnostics.close()
        if dashboard is not None:
            dashboard.stop()
        if kb is not None:
            kb.stop()
        source_selector.close()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
