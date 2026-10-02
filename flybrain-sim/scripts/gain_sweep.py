#!/usr/bin/env python3
"""
gain_sweep.py — Sweep FLOW_GAIN and record DNa02 response.

Injects constant asymmetric flow (fly heading slightly right, vx=5) at each
gain value, runs WARMUP then SAMPLE frames, and records mean DNa02 L/R spike
rates and nactive.

Biological target: ~0.5 spikes/window (26Hz bio × 0.020s tick).
Good operating range: 3–8 spikes/window (confirmed reliable steering).
Above ~150k nactive: gain too high, inhibitory cascade.

Output: results/gain_sweep_YYYY-MM-DD.csv

Usage:
    cd /path/to/flybrain-sim
    python3 scripts/gain_sweep.py [--gains 10,20,50,100,150,200,300,500]
"""

import argparse
import csv
import datetime
import gc
import json
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
    """CPU active-set reseed to driven neurons only."""
    if brain.nactive[0] > 0:
        brain.active_flag[brain.active[:brain.nactive[0]]] = 0
        brain.nactive[0] = 0
    driven = np.nonzero(drive)[0].astype(np.int32)
    n = len(driven)
    if n > 0:
        brain.active[:n] = driven
        brain.active_flag[driven] = 1
        brain.nactive[0] = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doomfly", default="../doomfly")
    ap.add_argument("--gains",   default="10,20,50,100,150,200,300,500",
                    help="Comma-separated FLOW_GAIN values")
    ap.add_argument("--warmup",  type=int, default=50,  help="Warmup frames per gain")
    ap.add_argument("--sample",  type=int, default=100, help="Sample frames per gain")
    ap.add_argument("--steps",   type=int, default=200, help="LIF ticks per frame")
    ap.add_argument("--out",     default=None,          help="Output CSV path")
    ap.add_argument("--cpu",     action="store_true",   help="Force CPU engine")
    args = ap.parse_args()

    gains = [float(g.strip()) for g in args.gains.split(",")]

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
    # Asymmetric flow: slight rightward heading → left eye gets more T4a signal
    # vx=5 gives a non-trivial T4a drive with good L/R contrast at FLOW_GAIN=150
    ENCODE_KWARGS = dict(x=W/2, y=H/2, vx=5.0, vy=0.0, heading=0.3,
                         world_width=W, world_height=H)

    out_path = Path(args.out) if args.out else (
        REPO_DIR / "results" / f"gain_sweep_{datetime.date.today()}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    norm = 200.0 / args.steps
    gc.disable()

    rows = []
    print(f"\n{'FLOW_GAIN':>10}  {'dna02_l':>8}  {'dna02_r':>8}  {'diff':>8}  {'nactive':>8}")
    print("-" * 52)

    for gain in gains:
        fe.FLOW_GAIN = gain
        drive = encoder.encode(**ENCODE_KWARGS)

        _reset_brain(brain, use_cuda)
        if not use_cuda:
            _reseed(brain, drive)

        # Warmup
        for _ in range(args.warmup):
            brain.drive[:] = drive
            brain.counts[:] = 0
            advance(brain, args.steps)

        # Sample
        l_vals, r_vals, na_vals = [], [], []
        for _ in range(args.sample):
            brain.drive[:] = drive
            brain.counts[:] = 0
            advance(brain, args.steps)
            l_vals.append(float(brain.counts[groups["dna02_left"]].mean()  * norm))
            r_vals.append(float(brain.counts[groups["dna02_right"]].mean() * norm))
            na_vals.append(int(brain.nactive[0]) if not use_cuda else 0)

        dna02_l = float(np.mean(l_vals))
        dna02_r = float(np.mean(r_vals))
        diff    = dna02_l - dna02_r
        nactive = float(np.mean(na_vals))

        print(f"{gain:>10.0f}  {dna02_l:>8.3f}  {dna02_r:>8.3f}  {diff:>+8.3f}  {nactive:>8.0f}")
        rows.append({
            "flow_gain":      gain,
            "dna02_l_mean":   round(dna02_l, 4),
            "dna02_r_mean":   round(dna02_r, 4),
            "diff_l_minus_r": round(diff, 4),
            "dna02_l_std":    round(float(np.std(l_vals)), 4),
            "dna02_r_std":    round(float(np.std(r_vals)), 4),
            "nactive_mean":   round(nactive, 0),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {out_path}")
    print("Interpretation:")
    print("  diff ≈ 0       → no L/R separation at this gain (gain too low or too high)")
    print("  diff stable    → gain in good operating window")
    print("  nactive >150k  → inhibitory cascade, gain too high")


if __name__ == "__main__":
    main()
