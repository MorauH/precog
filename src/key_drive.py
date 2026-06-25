#!/usr/bin/env python3
"""
teleop_keyboard.py — Keyboard → ROS2 drive command publisher

Keys:
  w / s  →  Torque  (forward / reverse)  → /cmd/trq   Float32MultiArray [FL, FR, RL, RR]
  a / d  →  Steering (left / right)      → /cmd/swa   Float32  [-1.0 … 1.0]
  q / Ctrl-C → quit

Values ramp smoothly toward ±1 while a key is held and decay back to 0
when released.  All rates are configurable at the top of the file.

Dependencies: none beyond ROS2 (rclpy, std_msgs) — uses stdlib tty/termios.
Works on Linux and WSL.
"""

import sys
import tty
import termios
import select
import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

# ── Tuning ───────────────────────────────────────────────────────────────────
UPDATE_HZ    = 100    # Control-loop frequency [Hz]
RAMP_RATE    = 1.5    # Units/second to ramp toward ±1 when key held
DECAY_RATE   = 2.0    # Units/second to decay toward 0 when key released
KEY_TIMEOUT  = 0.12   # Seconds without a repeat → key considered released
# ─────────────────────────────────────────────────────────────────────────────

# ANSI helpers
_CSI = "\033["
def _move_up(n):     return f"{_CSI}{n}A"
def _clear_line():   return f"{_CSI}2K\r"
def _bold(s):        return f"{_CSI}1m{s}{_CSI}0m"
def _dim(s):         return f"{_CSI}2m{s}{_CSI}0m"
def _color(s, c):    return f"{_CSI}{c}m{s}{_CSI}0m"

GREEN, YELLOW, RED, CYAN, WHITE = 32, 33, 31, 36, 37


def _bar(value, width=20):
    """Render a symmetric bar centred at 0."""
    half = width // 2
    pos  = min(int(abs(value) * half), half)
    if value >= 0:
        left  = " " * half
        right = "=" * pos + " " * (half - pos)
        arrow = ">" if value > 0.001 else "|"
    else:
        left  = " " * (half - pos) + "=" * pos
        right = " " * half
        arrow = "<" if value < -0.001 else "|"
    colour = GREEN if abs(value) < 0.4 else (YELLOW if abs(value) < 0.8 else RED)
    return _color(f"[{left}{arrow}{right}]", colour)


# ── Raw-terminal keyboard reader ─────────────────────────────────────────────

class KeyboardReader:
    """
    Reads single characters from stdin in raw mode (no Enter needed).
    Tracks the last-seen timestamp for each key so the caller can
    determine whether a key is currently held via is_held().
    """

    CONTROL_KEYS = {"\x03": "ctrl_c", "\x04": "ctrl_d"}

    def __init__(self, on_quit):
        self._last_seen: dict[str, float] = {}
        self._lock      = threading.Lock()
        self._on_quit   = on_quit
        self._old_attrs = termios.tcgetattr(sys.stdin)

        # Set raw mode so each keystroke arrives immediately
        tty.setcbreak(sys.stdin.fileno())

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                # Block up to 50 ms — keeps the thread responsive on exit
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    ch = sys.stdin.read(1)
                    ch = ch.lower()
                    if ch in self.CONTROL_KEYS or ch == "q":
                        self._on_quit()
                        return
                    with self._lock:
                        self._last_seen[ch] = time.monotonic()
        except Exception:
            self._on_quit()

    def is_held(self, key: str) -> bool:
        """True if the key was seen within KEY_TIMEOUT seconds."""
        with self._lock:
            t = self._last_seen.get(key)
        return t is not None and (time.monotonic() - t) < KEY_TIMEOUT

    def restore(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_attrs)


# ── ROS2 node ────────────────────────────────────────────────────────────────

class TeleopKeyboardNode(Node):

    def __init__(self):
        super().__init__("teleop_keyboard")

        # Publishers
        self.pub_swa = self.create_publisher(Float32,            "/cmd/steer", 10)
        self.pub_trq = self.create_publisher(Float32MultiArray,  "/cmd/trq", 10)
        self.pub_vel = self.create_publisher(Float32,            "/cmd/acc", 10)

        # Smoothed output values [-1, 1]
        self.steering = 0.0   # a → -1  |  d → +1
        self.torque   = 0.0   # s → -1  |  w → +1

        # Derived step sizes
        dt = 1.0 / UPDATE_HZ
        self._step_ramp  = RAMP_RATE  * dt
        self._step_decay = DECAY_RATE * dt
        self._running    = True

        # Keyboard (starts background thread)
        self._kbd = KeyboardReader(on_quit=self._request_shutdown)

        # Control/publish timer
        self.create_timer(dt, self._tick)

        self._print_header()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _request_shutdown(self):
        self._running = False

    def destroy_node(self):
        self._running = False
        self._kbd.restore()
        # Zero outputs on shutdown
        self.pub_swa.publish(Float32(data=0.0))
        trq = Float32MultiArray()
        trq.data = [0.0, 0.0, 0.0, 0.0]
        self.pub_trq.publish(trq)
        super().destroy_node()

    # ── Control loop ──────────────────────────────────────────────────────────

    def _tick(self):
        if not self._running:
            raise SystemExit

        a = self._kbd.is_held("a")
        d = self._kbd.is_held("d")
        w = self._kbd.is_held("w")
        s = self._kbd.is_held("s")

        # Steering  (a → negative, d → positive)
        if d and not a:
            self.steering = max(-0.3, self.steering - self._step_ramp)
        elif a and not d:
            self.steering = min(0.3,  self.steering + self._step_ramp)
        else:
            self.steering = self._decay(self.steering)

        # Torque  (s → negative, w → positive)
        if s and not w:
            self.torque = max(-1.0, self.torque - self._step_ramp)
        elif w and not s:
            self.torque = min(1.0,  self.torque + self._step_ramp)
        else:
            self.torque = self._decay(self.torque)

        # Publish steering
        self.pub_swa.publish(Float32(data=float(self.steering)))

        # Publish torque — same value on all four wheels [FL, FR, RL, RR]
        trq = Float32MultiArray()
        trq.data = [float(self.torque)] * 4
        self.pub_trq.publish(trq)
        
        self.pub_vel.publish(Float32(data=float(self.torque)))

        self._update_display(a, d, w, s)

    def _decay(self, value):
        if abs(value) <= self._step_decay:
            return 0.0
        return value - self._step_decay if value > 0 else value + self._step_decay

    # ── Terminal UI ───────────────────────────────────────────────────────────

    def _print_header(self):
        print()
        print(_bold("  ╔══════════════════════════════════════╗"))
        print(_bold("  ║     ROS2 Keyboard Teleop              ║"))
        print(_bold("  ╚══════════════════════════════════════╝"))
        print()
        print(_dim("  Controls:  W=forward  S=reverse  A=left  D=right  Q=quit"))
        print()
        # Reserve 4 lines for live display
        print("  Steering : ")
        print("  Torque   : ")
        print("  Keys     : ")
        print()

    _LIVE_LINES = 4

    def _update_display(self, a, d, w, s):
        key_map = {"W": w, "A": a, "S": s, "D": d}
        active  = [_color(k, CYAN) for k, held in key_map.items() if held]
        key_str = "  ".join(active) if active else _dim("(none)")

        sys.stdout.write(_move_up(self._LIVE_LINES))

        def line(label, bar, val):
            return f"{_clear_line()}  {_bold(label)} {bar}  {_color(f'{val:+.3f}', WHITE)}\n"

        sys.stdout.write(line("Steering :", _bar(self.steering), self.steering))
        sys.stdout.write(line("Torque   :", _bar(self.torque),   self.torque))
        sys.stdout.write(f"{_clear_line()}  Keys     : {key_str}\n")
        sys.stdout.write(f"{_clear_line()}\n")
        sys.stdout.flush()


# ── Entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboardNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        print("\n  Teleop stopped. Goodbye!\n")


if __name__ == "__main__":
    main()
