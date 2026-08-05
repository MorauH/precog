import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError(
        "Dashboard requires 'fastapi' and 'uvicorn'. "
        "Install with: uv pip install fastapi uvicorn[standard]"
    )


@dataclass
class Controls:
    """Thread-safe shared state between main loop and dashboard."""

    blend_steer: float = 0.0
    blend_acc: float = 0.0

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_blend_steer(self, value: float):
        with self._lock:
            self.blend_steer = max(0.0, min(1.0, float(value)))

    def set_blend_acc(self, value: float):
        with self._lock:
            self.blend_acc = max(0.0, min(1.0, float(value)))

    @property
    def blend_steer_safe(self) -> float:
        with self._lock:
            return self.blend_steer

    @property
    def blend_acc_safe(self) -> float:
        with self._lock:
            return self.blend_acc


class DashboardServer:
    """
    Web dashboard with real-time diagnostics and control.

    Usage from the main loop::

        dash = DashboardServer(port=8080)
        dash.start()

        while running:
            ...
            dash.update({
                "tick": tick_count,
                "tick_rate": hz,
                "mode": mode,
                "blend_ratio": source_selector.blend_ratio,
                "action": [steering, acceleration],
                "prev_action": [expert_steering, expert_acceleration],
                "source_selector": source_selector.snapshot(),
                "surprise": [s0, s1],
            })

        dash.stop()
    """

    def __init__(self, port: int = 8080):
        self._port = port
        self._app = FastAPI(title="precog dashboard")
        self._app.mount("/static", StaticFiles(directory=_static_dir()), name="static")

        self._controls = Controls()
        self._ws_clients: list[WebSocket] = []
        self._ws_lock = threading.Lock()

        self._latest: dict = {}
        self._latest_seq: int = 0
        self._data_lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._app.get("/")(self._index)
        self._app.websocket("/ws")(self._ws_handler)

    # ------------------------------------------------------------------ #
    # Public API (called from main thread)
    # ------------------------------------------------------------------ #

    @property
    def controls(self) -> Controls:
        return self._controls

    def update(self, data: dict[str, Any]):
        """Merge a diagnostics snapshot. Clients poll it from the WS loop."""
        data["ts"] = time.monotonic()
        with self._data_lock:
            self._latest.update(data)
            self._latest_seq += 1

    def watch_level_logs(self, num_levels: int, poll_interval_s: float = 0.5):
        """Spawn a background thread that polls LevelLog shm slots.

        Merges each level's diagnostics into the dashboard update dict
        under keys like ``level_0_surprise``, ``level_1_idle_time_us``, etc.

        Args:
            num_levels: Number of level log slots to watch (0 .. num_levels-1).
            poll_interval_s: Polling interval in seconds (default 0.5).
        """
        if not self._running:
            return

        def _poller():
            from precog.messaging import LevelLog

            logs: list[Optional[LevelLog]] = [None] * num_levels

            while self._running:
                merged: dict[str, Any] = {}
                for i in range(num_levels):
                    log = logs[i]
                    if log is None:
                        try:
                            log = LevelLog.attach(i)
                            logs[i] = log
                        except FileNotFoundError:
                            continue
                    snap = log.read()
                    if snap is None:
                        continue
                    lid = int(snap["level_idx"])
                    for key, val in snap.items():
                        if key == "level_idx":
                            continue
                        merged[f"level_{lid}_{key}"] = val
                if merged:
                    self.update(merged)
                time.sleep(poll_interval_s)

            for log in logs:
                if log is not None:
                    log.close()

        t = threading.Thread(target=_poller, daemon=True)
        t.start()

    def start(self):
        """Start the web server in a daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    # HTTP routes
    # ------------------------------------------------------------------ #

    async def _index(self):
        path = _static_dir() / "index.html"
        content = path.read_text()
        return HTMLResponse(content)

    # ------------------------------------------------------------------ #
    # WebSocket handler
    # ------------------------------------------------------------------ #

    async def _ws_handler(self, ws: WebSocket):
        await ws.accept()
        with self._ws_lock:
            self._ws_clients.append(ws)

        last_sent_seq = -1
        try:
            while self._running:
                # Non-blocking receive for control commands
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                    self._handle_command(raw)
                except (asyncio.TimeoutError, RuntimeError):
                    pass

                # Only push when new data is available
                with self._data_lock:
                    current_seq = self._latest_seq
                    current = dict(self._latest) if self._latest else {}
                if current_seq > last_sent_seq and current:
                    last_sent_seq = current_seq
                    try:
                        await ws.send_text(json.dumps(current, default=_json_fallback))
                    except Exception:
                        break
        except (WebSocketDisconnect, ConnectionError):
            pass
        finally:
            with self._ws_lock:
                if ws in self._ws_clients:
                    self._ws_clients.remove(ws)

    def _handle_command(self, raw: str):
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            return

        if "set_blend_steer" in cmd:
            self._controls.set_blend_steer(cmd["set_blend_steer"])
        if "set_blend_acc" in cmd:
            self._controls.set_blend_acc(cmd["set_blend_acc"])

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _serve(self):
        uvicorn.run(self._app, host="0.0.0.0", port=self._port, log_level="warning")


def _static_dir():
    import pathlib

    return pathlib.Path(__file__).resolve().parent / "static"


def _json_fallback(obj):
    return str(obj)
