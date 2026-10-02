"""
Definitive AN03A008 directional test.
CudaBrain spike ring is 19 slots. Clear it with 20 zero-drive frames after reset.
Then test directional response at FLOW_GAIN=0 + AN03A008_GAIN=2.0.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats

DOOMFLY  = Path("/home/mrmotts/fly-brain/doomfly")
FLYBRAIN = Path("/home/mrmotts/fly-brain/flybrain-sim-repo/flybrain-sim")
sys.path.insert(0, str(DOOMFLY))
sys.path.insert(0, str(FLYBRAIN))

from doom.engine import Brain
from engine_cuda import CudaBrain
from flow_encoder import FlowEncoder
import flow_encoder as fe
from brain_runner import load_groups, read_dn_rates

brain_cpu = Brain(DOOMFLY / "outputs/doom/malecns_v1/graph.npz")
brain = CudaBrain(brain_cpu)
groups = load_groups(FLYBRAIN / "neuron_groups.json", brain_cpu.ids)
encoder = FlowEncoder(groups, brain_cpu.n)
print("CUDA ready")

W, H = 1600, 1200
SPEED = 200.0

def run_clean(x, y, heading, fg=0, ag=2.0, n_flush=25, n_warm=5, n_avg=15):
    fe.FLOW_GAIN = fg; fe.LOOM_GAIN = 0; fe.AN03A008_GAIN = ag
    fe.MECH_GAIN = 0; fe.CONTACT_GAIN = 0
    # Reset + flush spike ring with zero drive
    brain.reset_state()
    brain.drive[:] = 0
    for _ in range(n_flush): brain.advance(50)
    # Now inject actual drive
    vx = SPEED * np.cos(heading); vy = SPEED * np.sin(heading)
    drive = encoder.encode(x=x, y=y, vx=vx, vy=vy, heading=heading,
                           world_width=W, world_height=H, margin=40.0)
    brain.drive[:] = drive
    for _ in range(n_warm): brain.advance(50)
    ls, rs = [], []
    for _ in range(n_avg):
        brain.drive[:] = drive; brain.advance(50)
        l, r = read_dn_rates(brain.counts, groups, 50)
        ls.append(l); rs.append(r)
    return np.mean(ls), np.mean(rs), float(drive[3305]), float(drive[2937])

# --- Verify: zero drive → zero DNa02 ---
print("\n=== Verify: pure zero drive (spike ring cleared) ===")
brain.reset_state()
brain.drive[:] = 0
for _ in range(25): brain.advance(50)
ls, rs = [], []
for _ in range(15):
    brain.drive[:] = 0; brain.advance(50)
    l, r = read_dn_rates(brain.counts, groups, 50)
    ls.append(l); rs.append(r)
print(f"  DNa02_L={np.mean(ls):.3f}  DNa02_R={np.mean(rs):.3f}")

# --- Main: FLOW_GAIN=0, AN03A008=2.0, directional test ---
print("\n=== Clean directional test (FLOW_GAIN=0, AN03A008_GAIN=2.0) ===")
print("pos                | AN03A_L | AN03A_R | DNa02_L | DNa02_R | diff   | dir_OK")
tests = [
    ("top_wall_east",  800,  80, 0.0,   "L>R"),
    ("bot_wall_east",  800, 1120, 0.0,  "R>L"),
    ("top_wall_west",  800,  80, np.pi, "R>L"),
    ("bot_wall_west",  800, 1120, np.pi,"L>R"),
    ("center_east",    800,  600, 0.0,  "≈0"),
]
for desc, x, y, heading, expected in tests:
    l, r, al, ar = run_clean(x, y, heading)
    diff = l - r
    ok = ""
    if expected == "L>R":   ok = "✓" if diff > 0.1 else "✗"
    elif expected == "R>L": ok = "✓" if diff < -0.1 else "✗"
    else: ok = "✓" if abs(diff) < 0.5 else "✗"
    print(f"{desc:20s} | {al:7.2f} | {ar:7.2f} | {l:.3f}   | {r:.3f}   | {diff:+.3f}  | {ok} (exp {expected})")

# --- Correlation: vary x along top wall ---
print("\n=== Correlation: x varies along top wall (y=80, heading east) ===")
drive_diffs, dna02_diffs = [], []
for xi in np.linspace(100, 1500, 20):
    l, r, al, ar = run_clean(int(xi), 80, 0.0)
    drive_diffs.append(al - ar)
    dna02_diffs.append(l - r)
if len(set(dna02_diffs)) > 1:
    rv, pv = stats.pearsonr(drive_diffs, dna02_diffs)
    print(f"r = {rv:.4f}  p = {pv:.4f}  n=20")
    print(f"drive_diff: min={min(drive_diffs):.1f} max={max(drive_diffs):.1f}")
    print(f"dna02_diff: min={min(dna02_diffs):.3f} max={max(dna02_diffs):.3f}")
else:
    print("DNa02_diff constant")
    print(f"Values: {set(dna02_diffs)}")
