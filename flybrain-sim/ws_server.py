"""
Asyncio WebSocket broadcaster.

Runs in a background thread. Main loop calls broadcast() with a dict;
all connected browser clients receive the latest payload.

Design: broadcast() only stores the latest message (no coroutine queue).
A single _send_loop task in the asyncio event loop sends it, skipping
stale queued frames. This prevents the coroutine backlog that causes
WS fps degradation over multi-minute runs.
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
        self._pending = threading.Event()  # set when new data is ready
        self._on_message = None

    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def broadcast(self, data: dict):
        """Thread-safe: update latest payload; send loop picks it up."""
        msg = json.dumps(data)
        with self._lock:
            self._latest = msg
        self._pending.set()

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
            await self._send_loop()

    async def _send_loop(self):
        """Single async task: sends latest message to all clients.
        Runs at most once per event loop tick; never accumulates a backlog."""
        loop = asyncio.get_event_loop()
        last_sent = None
        while True:
            # Wait for new data without blocking the event loop
            await loop.run_in_executor(None, self._pending.wait)
            self._pending.clear()
            with self._lock:
                msg = self._latest
            if msg != last_sent and self._clients:
                await self._send_all(msg)
                last_sent = msg

    def set_message_handler(self, fn):
        """Register a callback fn(dict) called on any browser → server message."""
        self._on_message = fn

    async def _handler(self, ws):
        self._clients.add(ws)
        try:
            with self._lock:
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
