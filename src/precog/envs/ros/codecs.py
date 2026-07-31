"""
Codec & transform registry: pluggable converters between ROS messages and arrays,
plus post-snapshot cross-key feature transforms.

Add support for a new ROS message type by registering ONE function here.
Nothing else in the project needs to know it exists - env_config.yaml
just references it by name.
"""

from typing import Callable, Dict
import numpy as np
import torch

# msg instance -> np.ndarray
OBS_CODECS: Dict[str, Callable] = {}
# (np.ndarray, msg_cls) -> msg instance
ACTION_CODECS: Dict[str, Callable] = {}
# snapshot dict -> modified snapshot dict (has access to ALL keys)
TRANSFORMS: Dict[str, Callable] = {}


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


def transform(name: str):
    def register(fn):
        TRANSFORMS[name] = fn
        return fn

    return register


# ---- Observation codecs (ROS msg -> np.ndarray) ----


@obs_codec("float")
def _float(msg) -> np.ndarray:
    return np.array([msg.data], dtype=np.float32)


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


@obs_codec("path")
def _path(msg) -> np.ndarray:
    return np.array(
        [[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float32
    )


@obs_codec("car_state")
def _car_state(msg) -> np.ndarray:
    lv = msg.linear_velocity
    av = msg.angular_velocity
    la = msg.linear_acceleration
    ori = msg.orientation
    return np.array(
        [
            lv.x,
            lv.y,
            lv.z,
            av.x,
            av.y,
            av.z,
            la.x,
            la.y,
            la.z,
            ori.x,
            ori.y,
            ori.z,
            ori.w,
        ],
        dtype=np.float32,
    )


@obs_codec("pose")
def _pose(msg) -> np.ndarray:
    pos = msg.position
    ori = msg.orientation
    return np.array(
        [
            pos.x,
            pos.y,
            ori.x,
            ori.y,
            ori.z,
            ori.w,
        ],
        dtype=np.float32,
    )


# ---- Action codecs (np.ndarray -> ROS msg) ----


@action_codec("float")
def _encode_float(value: np.float32, msg_cls):
    msg = msg_cls()
    msg.data = value
    return msg


@action_codec("float_array")
def _encode_float_array(value: np.ndarray, msg_cls):
    msg = msg_cls()
    msg.data = np.asarray(value, dtype=np.float32).flatten().tolist()
    return msg


# ---- Post-snapshot transforms (full dict -> full dict) ----
#
# Transforms run after _snapshot() builds the raw dict. They have access
# to ALL observation keys simultaneously, which is what you need when a
# feature depends on data from multiple topics (e.g. path waypoints
# relative to the car's own pose).
#
# A transform should return the snapshot dict (mutated in-place or replaced).


@transform("best_path_relative_sampling")
def _best_path_relative_sampling(snapshot: dict) -> dict:
    """Transform best_path waypoints from global frame to car-local frame.
    Then samples points at 0, 2, 5, 10, 15, 20, 30, 40, 50, meters ahead.
    """
    OUT_TOPIC = "best_path_relative_sampling"
    SAMPLE_DISTANCES = torch.tensor(
        [0, 2, 5, 10, 15, 20, 30, 40, 50], dtype=torch.float32
    )

    path = snapshot.get("best_path")
    car = snapshot.get("car_pose")
    if path is None or car is None:
        snapshot[OUT_TOPIC] = None
        return snapshot

    pos_x = car[0, 0]
    pos_y = car[0, 1]
    qx, qy, qz, qw = car[0, -4], car[0, -3], car[0, -2], car[0, -1]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    # --- Transform waypoint to local frame ---
    waypoints = path[0]  # [N, 2]
    dx = waypoints[:, 0] - pos_x
    dy = waypoints[:, 1] - pos_y
    cos_a, sin_a = torch.cos(yaw), torch.sin(yaw)
    rel_x = cos_a * dx + sin_a * dy
    rel_y = -sin_a * dx + cos_a * dy
    local_wp = torch.stack([rel_x, rel_y], dim=-1)  # [N, 2] with car at origin

    # --- Vectorised closest-point projection ---
    p0 = local_wp[:-1]  # [N-1, 2]
    p1 = local_wp[1:]  # [N-1, 2]
    seg = p1 - p0  # [N-1, 2]
    seg_len_sq = (seg * seg).sum(dim=-1)  # [N-1]
    seg_lengths = torch.sqrt(seg_len_sq.clamp(min=1e-12))  # [N-1]

    # Project origin (car) onto each segment; clamp t to [0, 1]
    t = ((-p0) * seg).sum(dim=-1) / seg_len_sq.clamp(min=1e-12)  # [N-1]
    t = t.clamp(0.0, 1.0)
    proj = p0 + t.unsqueeze(-1) * seg  # [N-1, 2]
    dists_to_car = torch.norm(proj, dim=-1)  # [N-1] (car is origin)

    best_seg = torch.argmin(dists_to_car)
    best_t = t[best_seg]

    # Arc-length table for the full polyline
    cum_lengths = torch.cat(
        [torch.zeros(1, device=local_wp.device), torch.cumsum(seg_lengths, dim=0)]
    )  # [N]

    # Arc length at the closest projected point
    start_arc = cum_lengths[best_seg] + best_t * seg_lengths[best_seg]

    # --- Arc-length interpolation at desired distances ---
    target_arcs = start_arc + SAMPLE_DISTANCES.to(local_wp.device)  # [S]

    # For each target arc length, find the segment and interpolate
    # searchsorted gives index i where cum_lengths[i-1] <= target < cum_lengths[i]
    idx = torch.searchsorted(cum_lengths, target_arcs) - 1
    idx = idx.clamp(0, len(seg_lengths) - 1)  # guard endpoints

    seg_t = (
        (target_arcs - cum_lengths[idx]) / seg_lengths[idx].clamp(min=1e-12)
    ).clamp(0.0, 1.0)  # [S]
    sampled = local_wp[idx] + seg_t.unsqueeze(-1) * seg[idx]  # [S, 2]

    snapshot[OUT_TOPIC] = sampled.reshape(1, -1)  # [1, 2*S] = [1, 18]
    return snapshot


@transform("lateral_deviation")
def _lateral_deviation(snapshot: dict) -> dict:
    """Extract lateral path deviation from best_path_relative_sampling.

    The first waypoint (0 m ahead) is where the car sits on the best
    path.  Its y-coordinate is the lateral offset from path centre.
    """
    OUT_TOPIC = "lateral_deviation"
    path = snapshot.get("best_path_relative_sampling")
    if path is None:
        snapshot[OUT_TOPIC] = None
        return snapshot
    # shape [1, 18] → slice y of first waypoint (index 1) → [1, 1]
    snapshot[OUT_TOPIC] = path[:, 1:2]
    return snapshot


@transform("normalize_observations")
def _normalize_observations(snapshot: dict) -> dict:
    """Apply fixed per-dimension scaling to keep model inputs ~[-1, 1].

    Scales are vehicle/track design parameters, not learned:
      - current_steering: actuator limit 0.5 rad
      - best_path_relative_sampling: x 50 m (lookahead), y 10 m (half track)
      - lateral_deviation: 10 m (half track width)
    """
    _SCALES = {
        "current_steering": [0.5],
        "best_path_relative_sampling": [
            50.0 if i % 2 == 0 else 10.0 for i in range(18)
        ],
        "lateral_deviation": [10.0],
    }

    for key, scales in _SCALES.items():
        value = snapshot.get(key)
        if value is None:
            continue
        s = torch.as_tensor(scales, device=value.device, dtype=value.dtype)
        snapshot[key] = value / s
    return snapshot
