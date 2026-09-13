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
def read_dn_rates(counts: np.ndarray, groups: dict):
    """
    Returns (left_rate, right_rate) in Hz-equivalent (spikes per 200 mini-ticks).
    Using DNg100 for forward drive, DNa02 for left/right differential.
    """
    fwd_l = counts[groups["dng100_left"]].mean()  if len(groups.get("dng100_left",  [])) else 0.0
    fwd_r = counts[groups["dng100_right"]].mean() if len(groups.get("dng100_right", [])) else 0.0
    turn_l = counts[groups["dna02_left"]].mean()  if len(groups.get("dna02_left",   [])) else 0.0
    turn_r = counts[groups["dna02_right"]].mean() if len(groups.get("dna02_right",  [])) else 0.0

    # Combine forward + turning into differential left/right drive
    forward = (fwd_l + fwd_r) / 2
    left_rate  = forward + turn_l
    right_rate = forward + turn_r
    return left_rate, right_rate

# ---- calibration helper ----
def run_calibration(brain, groups: dict, encoder, steps: int = 500):
    """
    Inject constant left-eye front-to-back flow, check DNa02 response.
    Expected: dna02_left ~26 Hz, dna02_right ~2 Hz at correct FLOW_GAIN.

    Runs CONTINUOUSLY — seeded once, no per-step reseed — so multi-hop
    propagation can build up across frames the same way the main loop does.
    Prints spike rates every 100 steps to show where the signal dies.
    """
    print("\n=== Calibration: constant left-eye forward flow ===")
    from flow_encoder import FLOW_GAIN
    print(f"FLOW_GAIN = {FLOW_GAIN}")

    brain.v[:] = -52
    brain.g[:] = 0
    brain.counts[:] = 0

    # Seed once — drive propagates continuously from here
    drive = encoder.encode(vx=5.0, vy=0.0, heading=0.0,
                           looming_left=0.0, looming_right=0.0)
    brain.drive[:] = drive
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
        brain.cursor = _advance(brain, 200)

        if i % 100 == 99 or i == 0:
            parts = []
            for key in TRACE_KEYS:
                arr = groups.get(key)
                if arr is not None and len(arr):
                    parts.append(f"{key}={brain.counts[arr].mean():.1f}")
            print(f"  step {i+1:>4}/{steps}  nactive={brain.nactive[0]:>6}  " + "  ".join(parts))

    print("\nTarget: dna02_left ~26, dna02_right ~2")
    print("Adjust FLOW_GAIN in flow_encoder.py if off.\n")

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
    args = parser.parse_args()

    doomfly_path = find_doomfly(args.doomfly)
    sys.path.insert(0, str(doomfly_path))

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

    groups = load_groups(Path(__file__).parent / "neuron_groups.json", brain.ids)
    encoder = FlowEncoder(groups, brain.n)

    world   = World(width=args.width, height=args.height)
    ws      = WsBroadcaster(port=8765)
    ws.start()
    print(f"WebSocket: ws://localhost:8765")

    if args.calibrate:
        print("First advance will JIT-compile Numba kernel (~30-60s) ...")
        run_calibration(brain, groups, encoder)
        # Continue into main loop after calibration

    # One-time seed: put T4/T5 driven neurons into the active set so the first
    # advance has something to propagate. After this we never wipe the active set.
    init_drive = encoder.encode(vx=1.0, vy=0.0, heading=0.0,
                                looming_left=0.0, looming_right=0.0)
    _reseed_driven(brain, init_drive)

    frame_dt = 1.0 / args.fps
    last_send = time.monotonic()
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

            # 2. Advance 20 × 0.1ms LIF ticks (state persists across frames)
            brain.counts[:] = 0
            if frame == 0:
                print("First main-loop advance (Numba JIT if not cached) ...")
            brain.cursor = _advance(brain, 20)
            if frame == 0:
                print(f"  Done. nactive={brain.nactive[0]}")

            # 3. Post-advance runaway guard: dampen + prune the active set.
            #    Must run AFTER advance (advance adds new neurons; dampening before
            #    doesn't prevent the set from growing during the advance itself).
            n = brain.nactive[0]
            if n > 15000:
                active_now = brain.active[:n]
                brain.v[active_now] *= 0.4
                # Drop neurons that are back near resting potential — they're
                # no longer contributing signal and just slow the next advance.
                still_on = active_now[brain.v[active_now] > -55.0]
                off      = active_now[brain.v[active_now] <= -55.0]
                brain.active_flag[off] = 0
                m = len(still_on)
                brain.active[:m] = still_on
                brain.nactive[0] = m

            # 3. Read motor output
            left_rate, right_rate = read_dn_rates(brain.counts, groups)

            # 4. Update world physics
            world.step(left_rate, right_rate, dt=frame_dt)

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

            frame += 1
            if frame % (args.fps * 5) == 0:
                rt = elapsed / frame_dt
                print(f"  frame {frame}  DN_L={left_rate:.1f}Hz DN_R={right_rate:.1f}Hz"
                      f"  spd={world.speed:.0f}px/s  pos=({world.x:.0f},{world.y:.0f})"
                      f"  rt={rt:.2f}x  nactive={brain.nactive[0]}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
