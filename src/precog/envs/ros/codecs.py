"""
Codec registry: pluggable converters between ROS messages and arrays.

Add support for a new ROS message type by registering ONE function here.
Nothing else in the project needs to know it exists - env_config.yaml
just references it by name.
"""

from typing import Callable, Dict
import numpy as np

# msg instance -> np.ndarray
OBS_CODECS: Dict[str, Callable] = {}
# (np.ndarray, msg_cls) -> msg instance
ACTION_CODECS: Dict[str, Callable] = {}


def obs_codec(name: str):
    def register(fn):
        OBS_CODECS[name] = fn
        return fn
    return register


def action_codec(name: str):
    def register(fn):
        ACTION_CODECS[name] = fn
        return fn
    return register


# ---- Observation codecs (ROS msg -> np.ndarray) ----

@obs_codec("point_xyz")
def _point_xyz(msg) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z], dtype=np.float32)


@obs_codec("quaternion_xyzw")
def _quat_xyzw(msg) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z, msg.w], dtype=np.float32)


@obs_codec("imu")
def _imu(msg) -> np.ndarray:
    a = msg.linear_acceleration
    g = msg.angular_velocity
    return np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float32)


@obs_codec("float_array")
def _float_array(msg) -> np.ndarray:
    return np.array(msg.data, dtype=np.float32)


# Example of what adding PointCloud2 support would look like - uncomment
# and adjust once you actually need it:
#
# @obs_codec("pointcloud_xyz")
# def _pointcloud_xyz(msg) -> np.ndarray:
#     import sensor_msgs_py.point_cloud2 as pc2
#     points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
#     return np.array(list(points), dtype=np.float32)


# ---- Action codecs (np.ndarray -> ROS msg) ----

@action_codec("float_array")
def _encode_float_array(value: np.ndarray, msg_cls):
    msg = msg_cls()
    msg.data = np.asarray(value, dtype=np.float32).flatten().tolist()
    return msg
