"""
Asyncio WebSocket broadcaster.

Runs in a background thread. Main loop calls broadcast() with a dict;
all connected browser clients receive it as JSON.
"""

import asyncio
import json
import threading
import websockets
import logging

log = logging.getLogger(__name__)


class WsBroadcaster:
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._latest: str = "{}"
        self._lock = threading.Lock()
        self._on_message = None

    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def broadcast(self, data: dict):
        """Thread-safe: send data to all connected clients."""
        msg = json.dumps(data)
        with self._lock:
            self._latest = msg
        if self._loop and self._clients:
            asyncio.run_coroutine_threadsafe(self._send_all(msg), self._loop)

    # ---- internal ----

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port,
                                    ping_interval=20, ping_timeout=10):
            log.info("WebSocket listening on ws://%s:%d", self.host, self.port)
            print(f"WebSocket: ws://localhost:{self.port}")
            await asyncio.Future()   # run forever

    def set_message_handler(self, fn):
        """Register a callback fn(dict) called on any browser → server message."""
        self._on_message = fn

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            await ws.send(self._latest)
            async for raw in ws:
                if self._on_message:
                    try:
                        self._on_message(json.loads(raw))
                    except Exception:
                        pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    async def _send_all(self, msg: str):
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead
