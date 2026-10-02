"""
Reseed experiment: compare DNa02 L/R differential
with and without per-frame active-set reseed on CPU engine.

Run: python3 reseed_experiment.py --frames 200 --no-looming --cpu
"""
import argparse
import sys
import csv
import os
from pathlib import Path
import numpy as np
from scipy import stats

# ---- path setup ----
SCRIPT_DIR = Path(__file__).resolve().parent
DOOMFLY = Path("/home/mrmotts/fly-brain/doomfly")
if not (DOOMFLY / "doom" / "engine.py").exists():
    print(f"ERROR: doomfly not found at {DOOMFLY}"); sys.exit(1)
sys.path.insert(0, str(DOOMFLY))
sys.path.insert(0, str(SCRIPT_DIR))

from doom.engine import Brain, advance
from flow_encoder import FlowEncoder
import flow_encoder as fe
from world import World
from brain_runner import load_groups, _reseed_driven, read_dn_rates

GROUPS_PATH = SCRIPT_DIR / "neuron_groups.json"
GRAPH_PATH  = DOOMFLY / "outputs/doom/malecns_v1/graph.npz"
STEPS = 50
N_FRAMES = 200
ARENA_W, ARENA_H = 1600, 1200


def run_condition(brain, groups, encoder, world, n_frames, use_reseed, label):
    """Run N frames. Returns list of (dna02_l, dna02_r, nactive) per frame."""
    # Reset world position
    world.x, world.y = world.width / 2.0, world.height / 2.0
    world.heading = 0.0
    world.vx, world.vy = 1.0, 0.0

    dt = 1.0 / 50.0  # 50fps
    results = []

    # Init active set
    drive = encoder.encode(
        x=world.x, y=world.y, vx=world.vx, vy=world.vy, heading=world.heading,
        world_width=world.width, world_height=world.height, margin=world.margin,
        mech_l=world.mech_left, mech_r=world.mech_right,
    )
    _reseed_driven(brain, drive)

    for frame in range(n_frames):
        drive = encoder.encode(
            x=world.x, y=world.y, vx=world.vx, vy=world.vy, heading=world.heading,
            world_width=world.width, world_height=world.height, margin=world.margin,
            mech_l=world.mech_left, mech_r=world.mech_right,
        )

        if use_reseed:
            _reseed_driven(brain, drive)

        brain.drive[:] = drive
        brain.counts[:] = 0
        advance(
            brain.ptr, brain.post, brain.weight,
            brain.v, brain.g, brain.refractory, brain.drive,
            brain.queue, brain.queue_count, brain.cursor,
            STEPS, brain.dt, brain.counts,
            brain.active, brain.active_flag, brain.nactive,
        )

        l, r = read_dn_rates(brain.counts, groups, STEPS)
        nact = int(brain.nactive[0])
        results.append((l, r, l - r, nact))

        world.step(dt, l, r, 0.0, 0.0, 0.0, 0.0)

    diffs = [x[2] for x in results]
    ls    = [x[0] for x in results]
    rs    = [x[1] for x in results]
    nacts = [x[3] for x in results]

    r_val = stats.pearsonr(ls, rs)[0] if len(set(ls)) > 1 else float('nan')
    print(f"\n=== {label} ===")
    print(f"  nactive: min={min(nacts)}  max={max(nacts)}  final={nacts[-1]}")
    print(f"  dna02_l: mean={np.mean(ls):.3f}  std={np.std(ls):.3f}")
    print(f"  dna02_r: mean={np.mean(rs):.3f}  std={np.std(rs):.3f}")
    print(f"  diff: mean={np.mean(diffs):.3f}  std={np.std(diffs):.3f}")
    print(f"  r(l,r)={r_val:.4f}")
    # cross-correlation diff with itself (autocorrelation at lag1)
    if len(diffs) > 2:
        ac, _ = stats.pearsonr(diffs[:-1], diffs[1:])
        print(f"  autocorr(diff, lag1)={ac:.4f}")
    return results


def main():
    print(f"Loading brain from {GRAPH_PATH} ...")
    brain = Brain(GRAPH_PATH)
    print(f"  {brain.n} neurons, {len(brain.post)} synapses")

    groups = load_groups(GROUPS_PATH, brain.ids)
    encoder = FlowEncoder(groups, brain.n)
    world = World(width=ARENA_W, height=ARENA_H)

    # CONDITION A: with per-frame reseed (proposed fix)
    res_with = run_condition(brain, groups, encoder, world,
                             N_FRAMES, use_reseed=True, label="WITH reseed (per-frame)")

    # CONDITION B: without per-frame reseed (current behavior)
    res_without = run_condition(brain, groups, encoder, world,
                                N_FRAMES, use_reseed=False, label="WITHOUT reseed (accumulates)")

    # Compare diffs
    diff_with    = [x[2] for x in res_with]
    diff_without = [x[2] for x in res_without]

    print("\n=== Comparison ===")
    r_cross, _ = stats.pearsonr(diff_with, diff_without)
    print(f"  r(diff_with, diff_without) = {r_cross:.4f}")

    # Save CSV
    out = SCRIPT_DIR / "results" / "reseed_experiment_2026-10-02.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame','reseed_l','reseed_r','reseed_diff','reseed_nactive',
                    'noreseed_l','noreseed_r','noreseed_diff','noreseed_nactive'])
        for i, (a, b) in enumerate(zip(res_with, res_without)):
            w.writerow([i, a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]])
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
