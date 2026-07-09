import rclpy
import torch
import yaml

from precog.envs import ROSEnvironment
from precog.model import DEFAULT_CONFIG, HierarchicalPCWorldModel
from precog.model.config import _env_shapes_from_yaml, resolve_config_dims


ENV_CONFIG_PATH = "./src/precog/envs/ros/env_config.yaml"


def main():
    rclpy.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Resolve model modality input_dim from env config shapes
    with open(ENV_CONFIG_PATH) as f:
        env_cfg = yaml.safe_load(f)
    env_shapes = _env_shapes_from_yaml(env_cfg)

    config = resolve_config_dims(DEFAULT_CONFIG, env_shapes)
    for m in config.modalities:
        print(f"  {m.name}: input_dim={m.input_dim} → output_dim={m.output_dim}")

    env = ROSEnvironment(config_path=ENV_CONFIG_PATH, device=device)
    model = HierarchicalPCWorldModel(config).to(device)

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=1.0,
        batch_size=1,
        device=device,
    )

    obs = env.reset()

    try:
        prev_action = None

        while rclpy.ok():
            result = runner.tick(obs, prev_action)
            obs = env.step(result.action)
            prev_action = result.action
            runner.clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
