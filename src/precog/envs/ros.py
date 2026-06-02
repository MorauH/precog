#!/usr/bin/env python3

import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

import torch
import numpy as np

# ROS 2 Message Types
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point, Quaternion


class ROSEnvironment(Node):
    """
    An OpenAI Gym-like interface wrapper for a physical robot using ROS 2 Jazzy.
    
    Controls: 'throttle' (Published to /robot/throttle)
    Sensors:  'position' (Subscribed to /robot/position)
              'orientation' (Subscribed to /robot/orientation)
    """

    def __init__(self, device: str = "cpu"):
        # Initialize the underlying ROS 2 Node
        super().__init__('ros_environment_wrapper')
        
        self.device = torch.device(device)
        self.state_lock = threading.Lock()
        self.first_obs_received = threading.Event()

        # ---------------------------------------------------------
        # 1. Internal Shared State (Observations)
        # ---------------------------------------------------------
        # Maps directly to the expected dictionary keys of the HierarchicalPCWorldModel
        self._latest_obs = {
            "position": None,     # Will be shape (1, 3) -> x, y, z
            "orientation": None   # Will be shape (1, 4) -> x, y, z, w
        }

        # ---------------------------------------------------------
        # 2. ROS 2 Publishers & Subscribers
        # ---------------------------------------------------------
        # Control Output (Throttle command)
        self.throttle_pub = self.create_publisher(
            Float32MultiArray, 
            '/robot/throttle', 
            10
        )

        # Sensor Inputs
        self.create_subscription(Point, '/robot/position', self._position_callback, 10)
        self.create_subscription(Quaternion, '/robot/orientation', self._orientation_callback, 10)

        # ---------------------------------------------------------
        # 3. Background Thread Spin Management
        # ---------------------------------------------------------
        # This keeps the subscribers active without blocking the main machine learning thread
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        
        self.get_logger().info("ROS Environment Wrapper initialized and spinning.")

    # -------------------------------------------------------------
    # Subscriber Callbacks (Asynchronous ROS Thread)
    # -------------------------------------------------------------
    def _position_callback(self, msg: Point):
        """Processes incoming Cartesian positions into a (1, 3) PyTorch tensor."""
        pos_array = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        pos_tensor = torch.as_tensor(pos_array, device=self.device).unsqueeze(0)
        
        with self.state_lock:
            self._latest_obs["position"] = pos_tensor
            self._check_initialization_status()

    def _orientation_callback(self, msg: Quaternion):
        """Processes incoming orientations (quaternions) into a (1, 4) PyTorch tensor."""
        ori_array = np.array([msg.x, msg.y, msg.z, msg.w], dtype=np.float32)
        ori_tensor = torch.as_tensor(ori_array, device=self.device).unsqueeze(0)
        
        with self.state_lock:
            self._latest_obs["orientation"] = ori_tensor
            self._check_initialization_status()

    def _check_initialization_status(self):
        """Helper to unlock the reset() block once data arrives across all channels."""
        if (self._latest_obs["position"] is not None and 
                self._latest_obs["orientation"] is not None):
            self.first_obs_received.set()

    # -------------------------------------------------------------
    # Public Gym-Style API (Main ML/Control Thread)
    # -------------------------------------------------------------
    def reset(self) -> tuple[dict[str, torch.Tensor], None]:
        """
        Blocks until the first valid ROS messages are received to 
        safely initialize the world model sequence.
        """
        self.get_logger().info("Waiting for initial sensor packets from ROS 2...")
        self.first_obs_received.wait()  # Block here until both callbacks fire at least once
        
        with self.state_lock:
            # Create a clean snapshot copies of the current dictionary
            obs_snapshot = {k: v.clone() for k, v in self._latest_obs.items()}
            
        self.get_logger().info("Sensors online. Environment reset complete.")
        return obs_snapshot, None

    def step(self, action_tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Executes a single control step.
        
        Parameters:
            action_tensor: PyTorch tensor containing the throttle sequence command.
        Returns:
            obs_dict: Thread-safe dictionary of the latest sensor states.
            action_tensor: Detached action echo acting as prev_action.
        """
        # 1. Publish the throttle action to the physical robot/simulation over ROS 2
        msg = Float32MultiArray()
        msg.data = action_tensor.squeeze(0).cpu().numpy().tolist()
        self.throttle_pub.publish(msg)

        # 2. Gather the latest observations using a thread lock to ensure thread consistency
        with self.state_lock:
            obs_snapshot = {k: v.clone() for k, v in self._latest_obs.items()}

        return obs_snapshot, action_tensor.detach()

    def close(self):
        """Shuts down the background thread and cleans up the node."""
        self._executor.shutdown()
        self.destroy_node()
        self._spin_thread.join()
