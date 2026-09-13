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
  --doomfly     Path to DOOMFLY repo (default: ../doomfly)
  --width/--height  World canvas size (must match fly.html)
"""

import argparse
import json
import sys
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
    p = Path(path).resolve()
    if not (p / "doom" / "engine.py").exists():
        print(f"ERROR: DOOMFLY not found at {p}")
        print("Clone it with: git clone https://github.com/nftechie/doomfly ../doomfly")
        sys.exit(1)
    return p

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

def _reset_state(brain, use_cuda, v=-52.0):
    if use_cuda:
        brain.reset_state(v=v)
    else:
        brain.v[:] = v
        brain.g[:] = 0
        brain.counts[:] = 0


# ---- calibration helper ----
def run_calibration(brain, groups: dict, encoder, steps: int = 500,
                    use_cuda: bool = False, do_advance=None):
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
    drive = encoder.encode(vx=5.0, vy=0.0, heading=0.0,
                           looming_left=0.0, looming_right=0.0)
    brain.drive[:] = drive
    if use_cuda:
        n_driven = int((drive != 0).sum())  # GPU runs all neurons; no active set
    else:
        n_driven = _reseed_driven(brain, drive)
    print(f"  Seeded {n_driven} driven neurons (continuous from here)")

    # Intermediate layers to trace signal depth
    TRACE_KEYS = [
        "t4a_left", "t4a_right",       # input layer
        "lc4_left", "lc4_right",        # looming / lobula
        "dna02_left", "dna02_right",    # target DNs
        "dng100_left", "dng100_right",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doomfly",   default="../doomfly")
    parser.add_argument("--width",     type=int, default=800)
    parser.add_argument("--height",    type=int, default=600)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--fps",       type=int, default=50)
    parser.add_argument("--steps",     type=int, default=50,
                        help="LIF ticks per frame (50=5ms fast, 200=20ms full; GPU makes 200 very slow)")
    parser.add_argument("--cpu",       action="store_true",
                        help="Force the CPU (numba njit) engine")
    parser.add_argument("--selftest",  action="store_true",
                        help="Run 200 steps on CPU+GPU and compare "
                             "per-neuron spike counts, then exit")
    args = parser.parse_args()

    doomfly_path = find_doomfly(args.doomfly)
    sys.path.insert(0, str(doomfly_path))

    if args.selftest:
        from engine_cuda import run_selftest
        run_selftest(doomfly_path, steps=200)
        return

    from doom.engine import Brain
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

    world   = World(width=args.width, height=args.height)
    ws      = WsBroadcaster(port=8765)
    ws.start()
    print(f"WebSocket: ws://localhost:8765")

    if args.calibrate:
        print("First advance will JIT-compile Numba kernel (~30-60s) ...")
        run_calibration(brain, groups, encoder, use_cuda=use_cuda,
                        do_advance=do_advance)
        # Continue into main loop after calibration

    # One-time seed: put T4/T5 driven neurons into the active set so the first
    # advance has something to propagate. After this we never wipe the active set.
    init_drive = encoder.encode(vx=1.0, vy=0.0, heading=0.0,
                                looming_left=0.0, looming_right=0.0)
    if not use_cuda:
        _reseed_driven(brain, init_drive)

    frame_dt = 1.0 / args.fps
    last_send = time.monotonic()
    actual_dt = frame_dt   # real wall-clock time per frame; updated each iteration
    frame = 0

    print(f"\nRunning at {args.fps} fps target. Open fly.html in browser.")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.monotonic()

            # 1. Encode synthetic optical flow → drive array
            drive = encoder.encode(
                world.vx, world.vy, world.heading,
                world.looming_left, world.looming_right,
            )
            brain.drive[:] = drive

            # 2. Advance 200 × 0.1ms LIF ticks = 20ms brain time per frame (matches 50fps).
            # 20 ticks (2ms) was too short: the 1.8ms synaptic delay alone is 18 ticks,
            # so spikes never reached downstream DNs and the fly ran on wander fallback.
            brain.counts[:] = 0
            if frame == 0:
                print("First main-loop advance (Numba JIT if not cached) ...")
            brain.cursor = do_advance(brain, args.steps)
            if frame == 0:
                print(f"  Done. nactive={brain.nactive[0]}")

            # No hard cap — let nactive stabilize naturally.
            # At FLOW_GAIN=20 the T4/T5 cascade should saturate within the visual
            # processing layers without blasting the whole 166k network.
            # If nactive still blooms to 166k, lower FLOW_GAIN further.

            # 3. Read motor output
            left_rate, right_rate = read_dn_rates(brain.counts, groups, args.steps)

            # 4. Update world physics — use real wall-clock dt so fly speed is
            # independent of GPU throughput (rt=7x was making it 7× too slow).
            world.step(left_rate, right_rate, dt=actual_dt)

            # 5. Broadcast to browser
            now = time.monotonic()
            if now - last_send >= frame_dt:
                state = world.state_dict()
                state.update({
                    "left_rate":  round(float(left_rate),  2),
                    "right_rate": round(float(right_rate), 2),
                    "frame": frame,
                })
                ws.broadcast(state)
                last_send = now

            # 6. Timing
            elapsed = time.monotonic() - t0
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
            actual_dt = max(elapsed, frame_dt)  # true wall time; feeds physics next frame

            frame += 1
            if frame % 10 == 0:   # print every 10 frames regardless of fps
                rt = elapsed / frame_dt
                print(f"  frame {frame}  DN_L={left_rate:.1f}Hz DN_R={right_rate:.1f}Hz"
                      f"  spd={world.speed:.0f}px/s  pos=({world.x:.0f},{world.y:.0f})"
                      f"  rt={rt:.2f}x  nactive={brain.nactive[0]}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
