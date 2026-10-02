#!/usr/bin/env python3
"""
ablation_6node.py — Test whether the 166k-neuron graph is a lookup table.

Runs the full connectome for N frames with naturalistic fly motion in a world
arena. At each frame records the T4a injection differential (input) and the
DNa02 spike-rate differential (output).

The key question: is out_diff = f(in_diff) a simple linear relationship?
  - Pearson r close to 1.0  → the graph is a linear passthrough (lookup table)
  - Lower r, or visible lag  → the graph adds nonlinearity / delay / cross-talk

Additionally computes cross-correlation at lags 0–5 to detect output delay.

Outputs:
  results/ablation_6node_YYYY-MM-DD.csv  — per-frame data
  (prints summary stats and correlation to stdout)

Usage:
    cd /path/to/flybrain-sim
    python3 scripts/ablation_6node.py [--frames 300] [--seed 42]
"""

import argparse
import csv
import datetime
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent


def _find_doomfly(hint: str) -> Path:
    candidates = [
        Path(hint).expanduser().resolve(),
        REPO_DIR / hint,
        REPO_DIR / "doomfly",
        REPO_DIR.parent / "doomfly",
        REPO_DIR.parent.parent / "doomfly",
    ]
    for p in candidates:
        p = p.resolve()
        if (p / "doom" / "engine.py").exists():
            return p
    print("ERROR: DOOMFLY not found. Try --doomfly /path/to/doomfly")
    sys.exit(1)


def _cuda_available() -> bool:
    try:
        import numba.cuda as _c
        return bool(_c.is_available())
    except Exception:
        return False


def _reset_brain(brain, use_cuda: bool):
    if use_cuda:
        brain.reset_state(v=-52.0)
    else:
        brain.v[:] = -52.0
        brain.g[:] = 0
    brain.counts[:] = 0


def _reseed(brain, drive: np.ndarray):
    if brain.nactive[0] > 0:
        brain.active_flag[brain.active[:brain.nactive[0]]] = 0
        brain.nactive[0] = 0
    driven = np.nonzero(drive)[0].astype(np.int32)
    n = len(driven)
    if n > 0:
        brain.active[:n] = driven
        brain.active_flag[driven] = 1
        brain.nactive[0] = n


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _crosscorr(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> dict:
    """Pearson r at lags 0..max_lag (b delayed relative to a)."""
    result = {}
    for lag in range(max_lag + 1):
        if lag == 0:
            result[lag] = _pearson(a, b)
        else:
            result[lag] = _pearson(a[:-lag], b[lag:])
    return result


def _simulate_world_step(world, fly_state, obstacles):
    """Minimal kinematic step — forward motion with simple wall bounce."""
    x, y, vx, vy, heading, speed = (
        fly_state["x"], fly_state["y"], fly_state["vx"], fly_state["vy"],
        fly_state["heading"], fly_state["speed"]
    )
    margin = 40.0
    # Gentle random heading drift
    heading += np.random.uniform(-0.05, 0.05)
    vx = math.cos(heading) * speed
    vy = math.sin(heading) * speed
    x += vx * 0.02
    y += vy * 0.02
    # Wall bounce
    if x < margin or x > world["w"] - margin:
        heading = math.pi - heading
        x = max(margin, min(world["w"] - margin, x))
    if y < margin or y > world["h"] - margin:
        heading = -heading
        y = max(margin, min(world["h"] - margin, y))
    fly_state.update(x=x, y=y, vx=vx, vy=vy, heading=heading)
    return fly_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doomfly", default="../doomfly")
    ap.add_argument("--frames",  type=int,   default=300, help="Frames to run")
    ap.add_argument("--warmup",  type=int,   default=50,  help="Warmup frames (discarded)")
    ap.add_argument("--steps",   type=int,   default=200, help="LIF ticks per frame")
    ap.add_argument("--seed",    type=int,   default=42,  help="RNG seed for reproducibility")
    ap.add_argument("--out",     default=None,            help="Output CSV path")
    ap.add_argument("--cpu",     action="store_true",     help="Force CPU engine")
    args = ap.parse_args()

    np.random.seed(args.seed)

    doomfly_path = _find_doomfly(args.doomfly)
    sys.path.insert(0, str(doomfly_path))
    sys.path.insert(0, str(REPO_DIR))

    from doom.engine import Brain
    import flow_encoder as fe
    from flow_encoder import FlowEncoder

    graph_path = doomfly_path / "outputs/doom/malecns_v1/graph.npz"
    if not graph_path.exists():
        print(f"ERROR: graph not found at {graph_path}")
        sys.exit(1)

    print(f"Loading brain … ", end="", flush=True)
    brain = Brain(graph_path)
    print(f"{brain.n} neurons, {len(brain.post)} synapses")

    use_cuda = False
    if not args.cpu:
        use_cuda = _cuda_available()
    if use_cuda:
        from engine_cuda import CudaBrain
        brain = CudaBrain(brain)
        print(f"  CUDA ({brain.device_bytes()/1e6:.0f} MB VRAM)")
    else:
        print("  CPU engine")

    def advance(b, steps):
        if use_cuda:
            return b.advance(steps)
        from doom.engine import advance as _adv
        return _adv(b.ptr, b.post, b.weight, b.v, b.g, b.refractory, b.drive,
                    b.queue, b.queue_count, b.cursor, steps, b.dt, b.counts,
                    b.active, b.active_flag, b.nactive)

    groups_path = REPO_DIR / "neuron_groups.json"
    raw = json.loads(groups_path.read_text())
    groups = {k: np.array(v, dtype=np.int32) for k, v in raw.items()}
    encoder = FlowEncoder(groups, brain.n)

    W, H = 1600, 1200
    # Seed world at centre heading east
    fly_state = {"x": W/2, "y": H/2, "vx": 5.0, "vy": 0.0,
                 "heading": 0.0, "speed": 5.0}
    world_meta = {"w": W, "h": H}
    obstacles = []  # no obstacles for this test — cleaner signal

    norm = 200.0 / args.steps
    gc.disable()
    _reset_brain(brain, use_cuda)

    # --- Warmup to steady network state ---
    print(f"Warmup ({args.warmup} frames) … ", end="", flush=True)
    for _ in range(args.warmup):
        fly_state = _simulate_world_step(world_meta, fly_state, obstacles)
        drive = encoder.encode(
            x=fly_state["x"], y=fly_state["y"],
            vx=fly_state["vx"], vy=fly_state["vy"],
            heading=fly_state["heading"],
            world_width=W, world_height=H,
        )
        brain.drive[:] = drive
        brain.counts[:] = 0
        if not use_cuda:
            _reseed(brain, drive)
        advance(brain, args.steps)
    print("done")

    # --- Data collection ---
    print(f"Collecting {args.frames} frames … ", end="", flush=True)

    rows = []
    t4a_l_sums, t4a_r_sums = [], []
    dna02_l_rates, dna02_r_rates = [], []

    for frame in range(args.frames):
        fly_state = _simulate_world_step(world_meta, fly_state, obstacles)
        drive = encoder.encode(
            x=fly_state["x"], y=fly_state["y"],
            vx=fly_state["vx"], vy=fly_state["vy"],
            heading=fly_state["heading"],
            world_width=W, world_height=H,
        )
        brain.drive[:] = drive
        brain.counts[:] = 0
        if not use_cuda:
            _reseed(brain, drive)
        advance(brain, args.steps)

        # T4a drive sum per hemifield (input signal)
        t4a_l = float(drive[groups["t4a_left"]].sum())
        t4a_r = float(drive[groups["t4a_right"]].sum())

        # DNa02 spike rate per hemifield (output signal)
        dn_l = float(brain.counts[groups["dna02_left"]].mean()  * norm)
        dn_r = float(brain.counts[groups["dna02_right"]].mean() * norm)

        # Also grab LPLC2 and DNp01 for cross-check
        lplc2_l = float(brain.counts[groups["lplc2_left"]].mean()  * norm)
        lplc2_r = float(brain.counts[groups["lplc2_right"]].mean() * norm)
        dnp01_l = float(brain.counts[groups["dnp01_left"]].mean()  * norm)
        dnp01_r = float(brain.counts[groups["dnp01_right"]].mean() * norm)

        nactive = int(brain.nactive[0]) if not use_cuda else 0

        t4a_l_sums.append(t4a_l)
        t4a_r_sums.append(t4a_r)
        dna02_l_rates.append(dn_l)
        dna02_r_rates.append(dn_r)

        rows.append({
            "frame":      frame,
            "fly_x":      round(fly_state["x"], 1),
            "fly_y":      round(fly_state["y"], 1),
            "heading":    round(fly_state["heading"], 4),
            "t4a_l_sum":  round(t4a_l, 4),
            "t4a_r_sum":  round(t4a_r, 4),
            "t4a_diff":   round(t4a_l - t4a_r, 4),
            "dna02_l":    round(dn_l, 4),
            "dna02_r":    round(dn_r, 4),
            "dna02_diff": round(dn_l - dn_r, 4),
            "lplc2_l":    round(lplc2_l, 4),
            "lplc2_r":    round(lplc2_r, 4),
            "dnp01_l":    round(dnp01_l, 4),
            "dnp01_r":    round(dnp01_r, 4),
            "nactive":    nactive,
        })

        if (frame + 1) % 50 == 0:
            print(f"{frame+1}", end=" ", flush=True)

    print("\ndone")

    # --- Analysis ---
    t4a_l_arr  = np.array(t4a_l_sums)
    t4a_r_arr  = np.array(t4a_r_sums)
    dn_l_arr   = np.array(dna02_l_rates)
    dn_r_arr   = np.array(dna02_r_rates)

    in_diff  = t4a_l_arr  - t4a_r_arr   # T4a L/R differential (input)
    out_diff = dn_l_arr   - dn_r_arr    # DNa02 L/R differential (output)

    r_diff = _pearson(in_diff, out_diff)
    r_l    = _pearson(t4a_l_arr, dn_l_arr)
    r_r    = _pearson(t4a_r_arr, dn_r_arr)

    lag_corr = _crosscorr(in_diff, out_diff, max_lag=5)

    print("\n=== Ablation Summary ===")
    print(f"Frames: {args.frames}  Steps/frame: {args.steps}  Seed: {args.seed}")
    print(f"\nDNa02 L/R differential vs T4a differential:")
    print(f"  Pearson r (diff vs diff)  = {r_diff:.4f}")
    print(f"  Pearson r (left channel)  = {r_l:.4f}")
    print(f"  Pearson r (right channel) = {r_r:.4f}")
    print(f"\nCross-correlation (output lagging input):")
    for lag, r in lag_corr.items():
        bar = "#" * int(abs(r) * 20)
        print(f"  lag={lag}: r={r:.4f}  {bar}")
    print(f"\nInterpretation:")
    if abs(r_diff) > 0.85:
        print("  r > 0.85 → GRAPH IS A LOOKUP TABLE. DNa02 tracks T4a linearly.")
        print("  The 166k connectome adds no asymmetric processing beyond direct drive.")
    elif abs(r_diff) > 0.5:
        print("  0.5 < r < 0.85 → MODERATE correlation. Graph may add nonlinearity.")
        print("  Check the cross-correlation: if lag>0 is highest, there is propagation delay.")
    else:
        print("  r < 0.5 → LOW correlation. The graph route differs from direct T4a→DNa02.")
        print("  This is the interesting case — the connectome is doing something.")

    best_lag = max(lag_corr, key=lambda k: abs(lag_corr[k]))
    if best_lag > 0:
        print(f"  Best correlation at lag={best_lag} frames → ~{best_lag * args.steps * 0.1:.0f}ms propagation delay")

    # --- Save CSV ---
    out_path = Path(args.out) if args.out else (
        REPO_DIR / "results" / f"ablation_6node_{datetime.date.today()}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRaw data saved: {out_path}")
    print("Columns: frame, fly_x/y, heading, t4a_l/r_sum, t4a_diff,")
    print("         dna02_l/r, dna02_diff, lplc2_l/r, dnp01_l/r, nactive")


if __name__ == "__main__":
    main()
