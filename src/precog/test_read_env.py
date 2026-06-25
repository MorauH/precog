import rclpy
import torch

from time import sleep

from precog.envs import ROSEnvironment

def main():
    rclpy.init()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    env = ROSEnvironment(config_path="./src/precog/envs/ros/env_config.yaml", device=device)

    obs = env.reset()

    try:
        while rclpy.ok():

            # Step the robot and get the updated telemetry
            obs = env.step([])

            print(obs["best_path_relative_sampling"])
            #for item in obs.values():
            #    print(item.shape)

            sleep(1/50)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
