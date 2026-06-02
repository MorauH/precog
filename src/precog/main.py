import rclpy
import torch

from precog.envs import ROSEnvironment
from precog.model import DEFAULT_CONFIG, HierarchicalPCWorldModel


def main():
    rclpy.init()

    # 1. Initialize environment and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    env = ROSEnvironment(device=device)

    model = HierarchicalPCWorldModel(DEFAULT_CONFIG).to(device)

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],  # Hz per level (index 0 = fastest)
        time_scale=1.0,
        batch_size=1,
        device="cpu",
    )

    # 2. Run standard control loop pattern
    obs, prev_action = env.reset()

    try:
        while rclpy.ok():
            # Tick world model
            result = runner.tick(obs, prev_action)

            # Step the robot and get the updated telemetry
            obs, prev_action = env.step(result.action)

            # Wait for the next 100 Hz wall-clock deadline
            runner.clock.sleep_until_next_tick()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
