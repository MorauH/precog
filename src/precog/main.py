import argparse
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
    if not args.headless:
        from precog.dashboard import DashboardServer

        dashboard = DashboardServer(port=args.dashboard_port)
        dashboard.start()
        print(f"  Dashboard: http://localhost:{args.dashboard_port}")

    obs = env.reset()

    print(f"\n  Starting")
    if dashboard:
        print(f"  Use the dashboard to control blend ratio")
    else:
        print(f"  Ctrl+C to quit")
    print()

    tick_count = 0
    blend_steer = 0.0
    blend_acc = 0.0

    try:
        while rclpy.ok():
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

            expert_action = _action_from_obs(obs, action_keys, device)

            if source_selector.last_output_valid:
                prev_action = torch.tensor(
                    [[source_selector.last_steer, source_selector.last_acc]],
                    dtype=torch.float32, device=device,
                )
            elif expert_action is not None:
                prev_action = expert_action
            else:
                prev_action = torch.zeros(1, config.control_dim, dtype=torch.float32, device=device)

            result = runner.tick(obs, prev_action, diagnostics=diagnostics)

            act_dict = _action_to_dict(result.action, action_keys)
            obs = env.step(act_dict)

            tick_count += 1

            if dashboard is not None and tick_count % 10 == 0:
                ma = result.action[0].tolist()
                pa = prev_action[0].tolist()
                ea = (
                    expert_action[0].tolist()
                    if expert_action is not None
                    else [None] * len(action_keys)
                )
                dashboard.update(
                    {
                        "tick": tick_count,
                        "tick_rate": diagnostics.tick_rate,
                        "blend_ratio_steer": dashboard.controls.blend_steer_safe,
                        "blend_ratio_acc": dashboard.controls.blend_acc_safe,
                        "action": ma,
                        "prev_action": pa,
                        "expert_action": ea,
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
                    pa = prev_action[0].tolist()
                    model_str = ", ".join(f"{v:.4f}" for v in ma)
                    prev_str = ", ".join(f"{v:.4f}" for v in pa)
                    ea_str = (
                        ", ".join(f"{v:.4f}" for v in expert_action[0].tolist())
                        if expert_action is not None
                        else "None"
                    )
                    print(
                        f"\n  tick={tick_count}  hz={hz:5.1f}  "
                        f"blend/s={blend_steer:.2f} blend/a={blend_acc:.2f}  "
                        f"model=[{model_str}]  prev=[{prev_str}]  expert=[{ea_str}]"
                    )
            elif diagnostics.should_report():
                ma = result.action[0].tolist()
                pa = prev_action[0].tolist()
                model_str = ", ".join(f"{v:.4f}" for v in ma)
                prev_str = ", ".join(f"{v:.4f}" for v in pa)
                print(
                    f"\n  tick={tick_count}  "
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
        source_selector.close()
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
