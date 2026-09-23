"""
Main loop: fly brain with synthetic optical flow input, WebSocket output.

Prerequisites:
  1. Clone DOOMFLY: git clone https://github.com/nftechie/doomfly ../doomfly
  2. Follow DOOMFLY setup (download MaleCNS data, run connectome/prepare/build_kernel)
  3. Run: python identify_neurons.py  (creates neuron_groups.json)
  4. Run: python brain_runner.py

Flags:
  --calibrate   Run calibration check: inject constant flow, print DNa02 fire rate.
                Target: ~26 Hz (excited side) vs ~2 Hz (other side).
  --no-looming  Zero escape DN input — removes loom turn and scatter (DNa02-only mode).
  --no-scatter  Disable bilateral-loom and corner scatter; keep directional loom turn.
  --doomfly     Path to DOOMFLY repo (default: ../doomfly)
  --width/--height  World canvas size (must match fly.html)

Threading: brain advances in a background thread (BrainWorker); physics runs
at full --fps using last-known DN rates between brain updates. Between updates
the fly reuses the previous rates — biologically reasonable since DN firing
persists between spike volleys. At 7x rt the brain updates ~28 times/sec
while physics runs at 50fps, giving smooth motion at the cost of ~35ms
latency in the neural feedback loop.
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np


def cuda_available() -> bool:
    try:
        import numba.cuda as _c
        return bool(_c.is_available())
    except Exception:
        return False

# ---- locate DOOMFLY ----
def find_doomfly(path: str) -> Path:
    """Resolve DOOMFLY from an explicit path or either repo layout.

    The simulator is commonly launched from the flybrain-sim directory, but
    users also launch it from the fly-brain repo root.  Resolve relative paths
    against both cwd and this file's parent so --doomfly is not required on
    every invocation.
    """
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path(__file__).resolve().parent / requested)
        candidates.append(Path(__file__).resolve().parent.parent / requested)
    # The default ../doomfly from flybrain-sim, and doomfly from repo root.
    candidates.extend([
        Path(__file__).resolve().parent.parent / "doomfly",
        Path(__file__).resolve().parent / ".." / "doomfly",
    ])
    seen = set()
    for candidate in candidates:
        p = candidate.resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / "doom" / "engine.py").exists():
            return p
    p = candidates[0].resolve()
    print(f"ERROR: DOOMFLY not found; checked:")
    for candidate in candidates:
        print(f"  {candidate.resolve()}")
    print("Clone it into fly-brain/doomfly or pass --doomfly /path/to/doomfly")
    sys.exit(1)

# ---- neuron group loading ----
def load_groups(groups_path: Path, brain_ids: np.ndarray) -> dict:
    if not groups_path.exists():
        print(f"ERROR: {groups_path} not found. Run identify_neurons.py first.")
        sys.exit(1)
    raw = json.loads(groups_path.read_text())
    return {k: np.array(v, dtype=np.int32) for k, v in raw.items()}

# ---- active set management ----
def _reseed_driven(brain, drive: np.ndarray):
    """Reset active set to only the neurons currently receiving drive.

    Called every frame. Prevents cascade accumulation: the DOOMFLY kernel
    adds all downstream partners to active regardless of whether they fire,
    so without a reset the active set grows to the full connectome within
    a few frames and the whole network ignites (inhibitory neurons suppress DNs).

    Resetting to just the driven neurons each frame lets the signal propagate
    fresh each 200-tick advance — T4a fires, chain propagates to DNs within
    those 200 ticks, then we reset for the next frame.
    """
    if brain.nactive[0] > 0:
        brain.active_flag[brain.active[:brain.nactive[0]]] = 0
        brain.nactive[0] = 0
    driven = np.nonzero(drive)[0].astype(np.int32)
    n = len(driven)
    if n > 0:
        brain.active[:n] = driven
        brain.active_flag[driven] = 1
        brain.nactive[0] = n
    return n

# ---- read DN fire rates ----
def read_dn_rates(counts: np.ndarray, groups: dict, steps: int = 200):
    """
    Returns (left_rate, right_rate) normalized to a 200-tick equivalent so
    SPEED_GAIN / TURN_GAIN in world.py are independent of --steps.

    e.g. 1 spike in 50 ticks → 4.0 (same as 4 spikes in 200 ticks).
    """
    norm = 200.0 / steps
    left_rate  = counts[groups["dna02_left"]].mean()  * norm if len(groups.get("dna02_left",  [])) else 0.0
    right_rate = counts[groups["dna02_right"]].mean() * norm if len(groups.get("dna02_right", [])) else 0.0
    return left_rate, right_rate

def read_group_rate(counts: np.ndarray, groups: dict, key: str, steps: int = 200) -> float:
    """Return a normalized mean spike count for an optional neuron group."""
    arr = groups.get(key)
    if arr is None or len(arr) == 0:
        return 0.0
    return float(counts[arr].mean() * (200.0 / steps))


def read_group_stats(counts: np.ndarray, groups: dict, key: str, steps: int = 200):
    """Return normalized (mean, max, active-neuron count) for diagnostics."""
    arr = groups.get(key)
    if arr is None or len(arr) == 0:
        return 0.0, 0.0, 0
    values = counts[arr].astype(np.float64) * (200.0 / steps)
    return float(values.mean()), float(values.max()), int(np.count_nonzero(values))


def read_looming_rates(counts: np.ndarray, groups: dict, steps: int = 200):
    """
    Returns (loom_left, loom_right) from LPLC2 spike counts, normalized to a
    200-tick equivalent.  Falls back to LC4 if LPLC2 indices are absent.

    These feed world.step() as the neural looming signal for escape turns.
    LC4/LPLC2 only fire significantly when the fly is within ~67px of a wall
    (looming input drives v_eq > threshold); outside that range the signal
    stays near the T4/T5-driven baseline and the threshold in world.py filters it out.
    """
    norm = 200.0 / steps
    loom_l = counts[groups["lplc2_left"]].mean()  * norm if len(groups.get("lplc2_left",  [])) else 0.0
    loom_r = counts[groups["lplc2_right"]].mean() * norm if len(groups.get("lplc2_right", [])) else 0.0
    if loom_l == 0 and loom_r == 0:
        loom_l = counts[groups["lc4_left"]].mean()  * norm if len(groups.get("lc4_left",  [])) else 0.0
        loom_r = counts[groups["lc4_right"]].mean() * norm if len(groups.get("lc4_right", [])) else 0.0
    return float(loom_l), float(loom_r)

def read_escape_dn_rates(counts: np.ndarray, groups: dict, steps: int = 200):
    """
    Returns (escape_left, escape_right) from DNp01 spike counts (normalized).
    Falls back to DNp103, then to LPLC2 if neither is in the graph.

    DNp01 is downstream of LPLC2 in the escape circuit (Ache et al. 2019);
    reading it is more honest than reading LPLC2 directly — the connectome
    has already integrated the looming signal through the escape pathway.
    """
    norm = 200.0 / steps
    for key_l, key_r in [("dnp01_left", "dnp01_right"),
                          ("dnp103_left", "dnp103_right"),
                          ("lplc2_left", "lplc2_right")]:
        nl = len(groups.get(key_l, []))
        nr = len(groups.get(key_r, []))
        if nl > 0 and nr > 0:
            el = float(counts[groups[key_l]].mean()  * norm)
            er = float(counts[groups[key_r]].mean() * norm)
            return el, er, key_l.split("_")[0]   # value, value, source name
    return 0.0, 0.0, "none"

def _reset_state(brain, use_cuda, v=-52.0):
    if use_cuda:
        brain.reset_state(v=v)
    else:
        brain.v[:] = v
        brain.g[:] = 0
        brain.counts[:] = 0


# ---- calibration helper ----
def run_calibration(brain, groups: dict, encoder, steps: int = 500,
                    use_cuda: bool = False, do_advance=None,
                    width: int = 800, height: int = 600):
    """
    Inject constant symmetric forward flow (both eyes), check DNa02 response.

    Each step runs 200 LIF ticks; counts are reset per step so the printout
    shows spikes-per-200-tick-window (not cumulative).

    NOTE on the boat.horse "26 Hz vs 2 Hz" target: that was for left-eye-only
    injection and measured in real Hz. In this simulation:
      - Both eyes are driven equally (forward flight, lat=0), so left≈right is correct.
      - 26 Hz biological = 26×0.020s = ~0.5 spikes/window for one neuron.
      - With FLOW_GAIN=20 we see ~6 spikes/window (≈300 Hz sim-rate); that is
        well above biology but adequate for navigation — DNa02 fires reliably
        and differential steering emerges naturally during turns.
      - Max physically possible: ~9 spikes/window (refractory = 22 ticks = 2.2ms).
      - Do NOT raise FLOW_GAIN trying to hit "26" counts — that target is wrong.

    Good calibration: dna02 stable at 3-8 spikes/window, nactive < 100k.
    """
    print("\n=== Calibration: constant left-eye forward flow ===")
    from flow_encoder import FLOW_GAIN
    print(f"FLOW_GAIN = {FLOW_GAIN}")

    _reset_state(brain, use_cuda)

    # Seed once — drive propagates continuously from here
    drive = encoder.encode(
        x=width / 2, y=height / 2, vx=5.0, vy=0.0, heading=0.0,
        world_width=width, world_height=height,
    )
    brain.drive[:] = drive
    if use_cuda:
        n_driven = int((drive != 0).sum())  # GPU runs all neurons; no active set
    else:
        n_driven = _reseed_driven(brain, drive)
    print(f"  Seeded {n_driven} driven neurons (continuous from here)")

    # Intermediate layers to trace signal depth
    TRACE_KEYS = [
        "t4a_left", "t4a_right",           # input layer
        "lc4_left", "lc4_right",           # looming / lobula
        "lplc2_left", "lplc2_right",       # looming escape sensor
        "dnp04_left", "dnp04_right",       # smooth looming avoidance DN (top LC4 target)
        "dnp01_left", "dnp01_right",       # escape command DN (downstream of LPLC2)
        "dnp103_left", "dnp103_right",     # escape command DN (highest LPLC2 weight)
        "dna02_left", "dna02_right",       # turn DNs
    ]

    for i in range(steps):
        brain.drive[:] = drive
        brain.counts[:] = 0
        brain.cursor = (do_advance or _advance)(brain, 200)

        if i % 100 == 99 or i == 0:
            parts = []
            for key in TRACE_KEYS:
                arr = groups.get(key)
                if arr is not None and len(arr):
                    parts.append(f"{key}={brain.counts[arr].mean():.1f}")
            print(f"  step {i+1:>4}/{steps}  nactive={brain.nactive[0]:>6}  " + "  ".join(parts))

    print("\nGood: dna02 stable at 3-8 spikes/window, nactive < 100k.")
    print("If dna02=0: lower FLOW_GAIN. If nactive>150k: lower FLOW_GAIN.\n")

def _advance(brain, steps: int):
    from doom.engine import advance
    return advance(
        brain.ptr, brain.post, brain.weight,
        brain.v, brain.g, brain.refractory, brain.drive,
        brain.queue, brain.queue_count, brain.cursor,
        steps, brain.dt, brain.counts,
        brain.active, brain.active_flag, brain.nactive,
    )


class BrainWorker:
    """Advances the connectome in a background thread.

    Physics loop reads last-computed rates each frame without blocking.
    Between brain updates the fly uses stale rates — biologically reasonable
    since DN firing persists between spike volleys.

    CUDA note: numba CUDA uses the primary device context which is shared
    across threads. All device arrays are written exclusively from this
    thread, so no concurrent kernel access occurs.
    """

    _CONTACT_DN_KEYS = (
        "dna02_left", "dna02_right", "dnp01_left", "dnp01_right",
        "dnp18_left", "dnp18_right", "dnp33_left", "dnp33_right",
        "dnp55_left", "dnp55_right", "dnp73_left", "dnp73_right",
        "dng29_left", "dng29_right", "dng99_left", "dng99_right",
    )

    def __init__(self, brain, groups: dict, do_advance, steps: int):
        self._brain   = brain
        self._groups  = groups
        self._advance = do_advance
        self._steps   = steps
        self._lock    = threading.Lock()
        self._pending = None
        self._rates   = self._zero_rates()
        self._running = False
        self._thread  = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def submit(self, drive: np.ndarray):
        """Hand off drive array; latest always wins — old pending is dropped."""
        with self._lock:
            self._pending = drive

    def rates(self) -> dict:
        """Non-blocking snapshot of the most recently computed rates."""
        with self._lock:
            return dict(self._rates)

    @staticmethod
    def _zero_rates() -> dict:
        return {
            "left": 0.0, "right": 0.0,
            "loom_l": 0.0, "loom_r": 0.0,
            "escape_l": 0.0, "escape_r": 0.0, "escape_src": "none",
            "dnp04_l": 0.0, "dnp04_r": 0.0,
            "mech_rate": 0.0, "mech_peak": 0.0, "mech_active": 0,
            "contact_dn_rates": {},
            "contact_dn_peak": 0.0,
            "contact_l": 0.0, "contact_r": 0.0,
            "nactive": 0,
        }

    def _loop(self):
        local_drive = None
        while self._running:
            with self._lock:
                if self._pending is not None:
                    local_drive = self._pending
                    self._pending = None
            if local_drive is None:
                time.sleep(0.001)
                continue

            self._brain.drive[:] = local_drive
            self._brain.counts[:] = 0
            self._advance(self._brain, self._steps)

            left_rate, right_rate      = read_dn_rates(self._brain.counts, self._groups, self._steps)
            loom_l, loom_r             = read_looming_rates(self._brain.counts, self._groups, self._steps)
            escape_l, escape_r, src    = read_escape_dn_rates(self._brain.counts, self._groups, self._steps)
            dnp04_l  = read_group_rate(self._brain.counts, self._groups, "dnp04_left",  self._steps)
            dnp04_r  = read_group_rate(self._brain.counts, self._groups, "dnp04_right", self._steps)
            mech_rate, mech_peak, mech_active = read_group_stats(
                self._brain.counts, self._groups, "mech_static", self._steps)
            cdn = {k: read_group_rate(self._brain.counts, self._groups, k, self._steps)
                   for k in self._CONTACT_DN_KEYS}

            with self._lock:
                self._rates = {
                    "left": left_rate,  "right": right_rate,
                    "loom_l": loom_l,   "loom_r": loom_r,
                    "escape_l": escape_l, "escape_r": escape_r, "escape_src": src,
                    "dnp04_l": dnp04_l, "dnp04_r": dnp04_r,
                    "mech_rate": mech_rate, "mech_peak": mech_peak, "mech_active": mech_active,
                    "contact_dn_rates": cdn,
                    "contact_dn_peak": max(cdn.values(), default=0.0),
                    "contact_l": cdn.get("dng29_left",  0.0),
                    "contact_r": cdn.get("dng29_right", 0.0),
                    "nactive": int(self._brain.nactive[0]),
                }


def _place_obstacles(world, n: int, r: int = 40, max_attempts: int = 200):
    """Place n non-overlapping circular obstacles inside the arena."""
    from world import World
    m = world.margin + r + 10   # keep centre away from walls
    placed = []
    attempts = 0
    while len(placed) < n and attempts < max_attempts:
        attempts += 1
        cx = np.random.uniform(m, world.width  - m)
        cy = np.random.uniform(m, world.height - m)
        # Reject if too close to an existing obstacle
        too_close = any(
            math.sqrt((cx - p["cx"])**2 + (cy - p["cy"])**2) < (r * 2 + world.margin)
            for p in placed
        )
        # Reject if inside the fly's starting area (centre ±80px)
        near_start = (abs(cx - world.x) < r + 80 and abs(cy - world.y) < r + 80)
        if not too_close and not near_start:
            placed.append({"cx": cx, "cy": cy, "r": float(r)})
    world.obstacles = placed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doomfly",   default="../doomfly")
    parser.add_argument("--width",     type=int, default=800)
    parser.add_argument("--height",    type=int, default=600)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--fps",       type=int, default=50)
    parser.add_argument("--steps",     type=int, default=50,
                        help="LIF ticks per frame (50=5ms fast, 200=20ms full; GPU makes 200 very slow)")
    parser.add_argument("--frames",    type=int, default=0,
                        help="Stop after this many frames (0 = run until Ctrl-C)")
    parser.add_argument("--mech-gain",    type=float, default=None,
                        help="Override mechanosensory (JO pool) injection gain for A/B tests")
    parser.add_argument("--contact-gain", type=float, default=None,
                        help="Override DNg29 direct injection gain (default: CONTACT_GAIN in flow_encoder.py)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Seed world randomness for comparable A/B runs")
    parser.add_argument("--print-dn",  action="store_true",
                        help="Print per-neuron candidate DN rates on contact frames")
    parser.add_argument("--mechanosensory-only", action="store_true",
                        help="Disable T4/T5 and LC4/LPLC2 visual input; keep contact input")
    parser.add_argument("--no-mechanosensory", action="store_true",
                        help="Ablate contact injection while retaining the same world/contact timing")
    parser.add_argument("--no-looming", action="store_true",
                        help="Zero out escape DN input to world — removes loom turn and scatter; DNa02-only mode")
    parser.add_argument("--no-scatter", action="store_true",
                        help="Disable bilateral-loom and corner-contact scatter kicks; keep directional loom turn")
    parser.add_argument("--obstacles",  type=int, default=0,
                        help="Place N circular obstacles in the arena (r=40px each)")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force the CPU (numba njit) engine")
    parser.add_argument("--selftest",  action="store_true",
                        help="Run 200 steps on CPU+GPU and compare "
                             "per-neuron spike counts, then exit")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    doomfly_path = find_doomfly(args.doomfly)
    sys.path.insert(0, str(doomfly_path))

    if args.selftest:
        from engine_cuda import run_selftest
        run_selftest(doomfly_path, steps=200)
        return

    from doom.engine import Brain
    import flow_encoder
    from flow_encoder import FlowEncoder
    from world import World
    from ws_server import WsBroadcaster

    graph_path = doomfly_path / "outputs/doom/malecns_v1/graph.npz"
    if not graph_path.exists():
        print(f"ERROR: graph not found at {graph_path}")
        print("Complete DOOMFLY's setup pipeline first.")
        sys.exit(1)

    print(f"Loading brain from {graph_path} ...")
    brain = Brain(graph_path)
    print(f"  {brain.n} neurons, {len(brain.post)} synapses")

    # ---- backend selection: GPU if available unless --cpu ----
    use_cuda = False
    if not args.cpu:
        use_cuda = cuda_available()
        if not use_cuda:
            print("WARNING: numba.cuda.is_available() is False — "
                  "falling back to the CPU engine.")
    else:
        print("--cpu: forcing CPU engine.")
    if use_cuda:
        from engine_cuda import CudaBrain
        brain = CudaBrain(brain)
        print(f"  CUDA engine ready ({brain.device_bytes()/1e6:.0f} MB VRAM)")

    def do_advance(brain, steps):
        if use_cuda:
            return brain.advance(steps)
        return _advance(brain, steps)

    groups = load_groups(Path(__file__).parent / "neuron_groups.json", brain.ids)
    encoder = FlowEncoder(groups, brain.n)

    if args.mech_gain is not None:
        flow_encoder.MECH_GAIN = float(args.mech_gain)
        print(f"Mechanosensory gain override: {flow_encoder.MECH_GAIN:.2f}")
    if args.contact_gain is not None:
        flow_encoder.CONTACT_GAIN = float(args.contact_gain)
        print(f"Contact (DNg29) gain override: {flow_encoder.CONTACT_GAIN:.2f}")
    if args.no_mechanosensory:
        print("Mechanosensory ablation: contact detector retained, injection disabled")
    if args.mechanosensory_only:
        # Keep the real contact encoder active, but remove the visual input
        # channels. This makes downstream activity attributable to JO-C/E
        # input rather than mixed T4/T5 and looming drive.
        flow_encoder.FLOW_GAIN = 0.0
        flow_encoder.LOOM_GAIN = 0.0
        print("Mechanosensory-only mode: visual and looming drive disabled")

    world   = World(width=args.width, height=args.height)
    if args.obstacles > 0:
        _place_obstacles(world, args.obstacles)
        print(f"Obstacles: {len(world.obstacles)} circles placed")
    ws      = WsBroadcaster(port=8765)

    def handle_browser_msg(msg: dict):
        if "turn_gain" in msg:
            world.turn_gain = float(msg["turn_gain"])
            print(f"  [ui] turn_gain → {world.turn_gain:.2f}")

    ws.set_message_handler(handle_browser_msg)
    ws.start()
    print(f"WebSocket: ws://localhost:8765")

    if args.calibrate and args.mechanosensory_only:
        print("WARNING: --calibrate is visual calibration and is disabled in mechanosensory-only mode.")
    if args.calibrate and not args.mechanosensory_only:
        print("First advance will JIT-compile Numba kernel (~30-60s) ...")
        if not args.mechanosensory_only:
            run_calibration(brain, groups, encoder, use_cuda=use_cuda,
                            do_advance=do_advance,
                            width=args.width, height=args.height)
        # Continue into main loop after calibration

    # One-time seed for CPU active-set management; CUDA ignores it.
    init_drive = encoder.encode(
        x=world.x, y=world.y, vx=1.0, vy=0.0, heading=world.heading,
        world_width=world.width, world_height=world.height, margin=world.margin,
        mech_l=world.mech_left, mech_r=world.mech_right,
    )
    if not use_cuda:
        _reseed_driven(brain, init_drive)

    # Start brain worker thread — advances connectome in background so the
    # physics loop runs at full fps without waiting for the GPU.
    print("Starting brain worker thread ...")
    worker = BrainWorker(brain, groups, do_advance, args.steps)
    worker.submit(init_drive)
    worker.start()

    frame_dt = 1.0 / args.fps
    last_send = time.monotonic()
    actual_dt = frame_dt
    frame = 0

    print(f"\nRunning at {args.fps} fps target (brain in background). Open fly.html.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.monotonic()

            # 1. Encode current world state and hand off to brain worker.
            mech_input_l = 0.0 if args.no_mechanosensory else world.mech_left
            mech_input_r = 0.0 if args.no_mechanosensory else world.mech_right
            drive = encoder.encode(
                world.x, world.y, world.vx, world.vy, world.heading,
                world.width, world.height, world.margin, world.obstacles,
                mech_l=mech_input_l, mech_r=mech_input_r,
            )
            worker.submit(drive)

            # 2. Read last-known DN rates — non-blocking, no brain stall.
            r = worker.rates()
            left_rate        = r["left"]
            right_rate       = r["right"]
            loom_l           = r["loom_l"]
            loom_r           = r["loom_r"]
            escape_l         = r["escape_l"]
            escape_r         = r["escape_r"]
            escape_src       = r["escape_src"]
            dnp04_l          = r["dnp04_l"]
            dnp04_r          = r["dnp04_r"]
            mech_rate        = r["mech_rate"]
            mech_peak        = r["mech_peak"]
            mech_active      = r["mech_active"]
            contact_dn_rates = r["contact_dn_rates"]
            contact_dn_peak  = r["contact_dn_peak"]
            contact_l        = r["contact_l"]
            contact_r        = r["contact_r"]
            nactive          = r["nactive"]

            if args.print_dn and (mech_input_l > 0.0 or mech_input_r > 0.0):
                compact_dn = " ".join(
                    f"{key}={value:.1f}" for key, value in contact_dn_rates.items()
                    if value > 0.0
                ) or "none"
                print(f"  CONTACT_DN frame={frame} "
                      f"mech_in={mech_input_l:.2f}/{mech_input_r:.2f} {compact_dn}")

            # 3. Physics step using last-known rates — runs every frame at full fps.
            loom_l_in = 0.0 if args.no_looming else escape_l
            loom_r_in = 0.0 if args.no_looming else escape_r
            world.step(left_rate, right_rate, dt=actual_dt,
                       loom_l=loom_l_in, loom_r=loom_r_in,
                       contact_l=contact_l, contact_r=contact_r,
                       no_scatter=args.no_scatter,
                       dnp04_l=dnp04_l, dnp04_r=dnp04_r)

            # 4. Broadcast to browser
            now = time.monotonic()
            if now - last_send >= frame_dt:
                state = world.state_dict()
                state.update({
                    "left_rate":  round(float(left_rate),  2),
                    "right_rate": round(float(right_rate), 2),
                    "mech_left":  round(float(mech_input_l), 3),
                    "mech_right": round(float(mech_input_r), 3),
                    "mech_rate":  round(mech_rate, 2),
                    "mech_peak":  round(mech_peak, 2),
                    "mech_active": mech_active,
                    "contact_dn":  contact_dn_rates,
                    "contact_dn_peak": round(contact_dn_peak, 2),
                    "frame": frame,
                })
                ws.broadcast(state)
                last_send = now

            # 5. Timing — physics frame rate; brain runs independently in background
            elapsed = time.monotonic() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
            actual_dt = max(elapsed, frame_dt)

            frame += 1
            if args.frames and frame >= args.frames:
                print(f"Completed requested {args.frames} frames.")
                break
            if frame % 10 == 0:
                rt = elapsed / frame_dt
                print(f"  frame {frame}  DN_L={left_rate:.1f} DN_R={right_rate:.1f}"
                      f"  esc_L={escape_l:.1f} esc_R={escape_r:.1f} [{escape_src}]"
                      f"  p04_L={dnp04_l:.1f} p04_R={dnp04_r:.1f}"
                      f"  loom_L={loom_l:.1f} loom_R={loom_r:.1f}"
                      f"  mech_in={mech_input_l:.2f}/{mech_input_r:.2f}"
                      f" mech_spk={mech_rate:.2f}"
                      f" mech_peak={mech_peak:.1f} active={mech_active}"
                      f" dn_peak={contact_dn_peak:.1f}"
                      f"  turn_dn={world.last_dn_turn:+.2f}"
                      f"  turn_loom={world.last_loom_turn:+.2f}"
                      f"  turn_p04={world.last_dnp04_turn:+.2f}"
                      f"  turn_contact={world.last_contact_turn:+.2f}"
                      f"  scatter={world.last_scatter_turn:+.2f}"
                      f"  spd={world.speed:.0f}px/s  pos=({world.x:.0f},{world.y:.0f})"
                      f"  rt={rt:.2f}x  nactive={nactive}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
