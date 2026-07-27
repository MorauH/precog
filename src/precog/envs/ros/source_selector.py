import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32


BLEND_HZ = 100.0
STALE_TIMEOUT_SEC = 0.5


def _sub_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _pub_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
    )


class SourceSelector(Node):
    """
    Blends control commands from the external main algorithm (``*_auto``
    topics) and precog (``*_precog`` topics), publishing the result to
    the vehicle's actual control topics.

    Subscriptions
    -------------
    /cmd/steer_auto    (Float32)
    /cmd/acc_auto      (Float32)
    /cmd/steer_precog  (Float32)
    /cmd/acc_precog    (Float32)

    Publications
    ------------
    /cmd/steer         (Float32)  — blended steering
    /cmd/acc           (Float32)  — blended acceleration

    ``blend_ratio`` is thread-safe: 0.0 = full auto, 1.0 = full precog.
    """

    def __init__(self):
        super().__init__("source_selector")

        self._lock = threading.Lock()
        self._blend_ratio_steer: float = 0.0
        self._blend_ratio_acc: float = 0.0

        self._auto_steer: Optional[float] = None
        self._auto_acc: Optional[float] = None
        self._precog_steer: Optional[float] = None
        self._precog_acc: Optional[float] = None

        self._auto_steer_ts: float = 0.0
        self._auto_acc_ts: float = 0.0
        self._precog_steer_ts: float = 0.0
        self._precog_acc_ts: float = 0.0

        sub_qos = _sub_qos()
        pub_qos = _pub_qos()

        self.create_subscription(Float32, "/cmd/steer_auto", self._cb_auto_steer, sub_qos)
        self.create_subscription(Float32, "/cmd/acc_auto", self._cb_auto_acc, sub_qos)
        self.create_subscription(
            Float32, "/cmd/steer_precog", self._cb_precog_steer, sub_qos
        )
        self.create_subscription(Float32, "/cmd/acc_precog", self._cb_precog_acc, sub_qos)

        self._pub_steer = self.create_publisher(Float32, "/cmd/steer", pub_qos)
        self._pub_acc = self.create_publisher(Float32, "/cmd/acc", pub_qos)

        dt = 1.0 / BLEND_HZ
        self._timer = self.create_timer(dt, self._blend_and_publish)

        self._log_interval = 2.0
        self._last_log = time.monotonic()
        self._stale_warned: set[str] = set()

        self._spin_executor = SingleThreadedExecutor()
        self._spin_executor.add_node(self)
        self._bg_thread = threading.Thread(target=self._spin_executor.spin, daemon=True)
        self._bg_thread.start()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def blend_ratio_steer(self) -> float:
        with self._lock:
            return self._blend_ratio_steer

    @blend_ratio_steer.setter
    def blend_ratio_steer(self, value: float):
        with self._lock:
            self._blend_ratio_steer = max(0.0, min(1.0, float(value)))

    @property
    def blend_ratio_acc(self) -> float:
        with self._lock:
            return self._blend_ratio_acc

    @blend_ratio_acc.setter
    def blend_ratio_acc(self, value: float):
        with self._lock:
            self._blend_ratio_acc = max(0.0, min(1.0, float(value)))

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "blend_ratio_steer": self._blend_ratio_steer,
                "blend_ratio_acc": self._blend_ratio_acc,
                "auto_steer": self._auto_steer,
                "auto_acc": self._auto_acc,
                "precog_steer": self._precog_steer,
                "precog_acc": self._precog_acc,
                "stale": sorted(self._stale_warned),
            }

    def close(self):
        self._spin_executor.shutdown()
        self.destroy_node()

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _cb_auto_steer(self, msg: Float32):
        with self._lock:
            self._auto_steer = msg.data
            self._auto_steer_ts = time.monotonic()

    def _cb_auto_acc(self, msg: Float32):
        with self._lock:
            self._auto_acc = msg.data
            self._auto_acc_ts = time.monotonic()

    def _cb_precog_steer(self, msg: Float32):
        with self._lock:
            self._precog_steer = msg.data
            self._precog_steer_ts = time.monotonic()

    def _cb_precog_acc(self, msg: Float32):
        with self._lock:
            self._precog_acc = msg.data
            self._precog_acc_ts = time.monotonic()

    # ------------------------------------------------------------------ #
    # Blend & publish
    # ------------------------------------------------------------------ #

    def _blend_and_publish(self):
        now = time.monotonic()
        with self._lock:
            ratio_s = self._blend_ratio_steer
            ratio_a = self._blend_ratio_acc
            auto_s = self._auto_steer
            auto_a = self._auto_acc
            prec_s = self._precog_steer
            prec_a = self._precog_acc
            auto_s_ts = self._auto_steer_ts
            auto_a_ts = self._auto_acc_ts
            prec_s_ts = self._precog_steer_ts
            prec_a_ts = self._precog_acc_ts

        stale: list[str] = []

        # Steering
        if prec_s is None:
            steer_out = auto_s if auto_s is not None else 0.0
        elif auto_s is None:
            steer_out = prec_s
        else:
            steer_out = ratio_s * prec_s + (1.0 - ratio_s) * auto_s
        self._pub_steer.publish(Float32(data=float(steer_out)))

        # Acceleration
        if prec_a is None:
            acc_out = auto_a if auto_a is not None else 0.0
        elif auto_a is None:
            acc_out = prec_a
        else:
            acc_out = ratio_a * prec_a + (1.0 - ratio_a) * auto_a
        self._pub_acc.publish(Float32(data=float(acc_out)))

        # Staleness check (log only)
        def _check(label: str, ts: float):
            if ts > 0 and now - ts > STALE_TIMEOUT_SEC:
                stale.append(label)

        _check("steer_auto", auto_s_ts)
        _check("acc_auto", auto_a_ts)
        _check("steer_precog", prec_s_ts)
        _check("acc_precog", prec_a_ts)

        with self._lock:
            newly_stale = set(stale) - self._stale_warned
            no_longer_stale = self._stale_warned - set(stale)
            self._stale_warned = set(stale)

        if newly_stale and now - self._last_log > self._log_interval:
            self.get_logger().warn(f"Stale sources: {sorted(newly_stale)}")
            self._last_log = now
        if no_longer_stale and now - self._last_log > self._log_interval:
            self.get_logger().info(f"Sources recovered: {sorted(no_longer_stale)}")
            self._last_log = now
