"""AN03A008 CUDA gain test with proper brain state reset between cases."""
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

GRAPH_PATH  = DOOMFLY / "outputs/doom/malecns_v1/graph.npz"
GROUPS_PATH = FLYBRAIN / "neuron_groups.json"
STEPS = 50
W, H = 1600, 1200
SPEED = 200.0

print("Loading brain (CUDA)...")
brain_cpu = Brain(GRAPH_PATH)
brain = CudaBrain(brain_cpu)
groups = load_groups(GROUPS_PATH, brain_cpu.ids)
encoder = FlowEncoder(groups, brain_cpu.n)

def run_frame(x, y, heading, speed=SPEED, n_warmup=3, n_avg=5):
    """Reset state, warm up n_warmup frames, average over n_avg frames."""
    brain.reset_state()
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    drive = encoder.encode(x=x, y=y, vx=vx, vy=vy, heading=heading,
                           world_width=W, world_height=H, margin=40.0)
    brain.drive[:] = drive
    # Warm up
    for _ in range(n_warmup):
        brain.drive[:] = drive
        brain.advance(STEPS)
    # Average
    ls, rs = [], []
    for _ in range(n_avg):
        brain.drive[:] = drive
        brain.advance(STEPS)
        l, r = read_dn_rates(brain.counts, groups, STEPS)
        ls.append(l); rs.append(r)
    al = float(drive[3305]); ar = float(drive[2937])
    return np.mean(ls), np.mean(rs), al, ar

# --- Gain sweep at top-wall asymmetric position ---
print("\n=== GAIN sweep (top wall y=80, heading east, speed=200) ===")
print("GAIN  | AN03A_L | AN03A_R | DNa02_L | DNa02_R | diff")
for gain in [0.2, 0.5, 1.0, 2.0, 5.0]:
    fe.AN03A008_GAIN = gain
    l, r, al, ar = run_frame(800, 80, 0.0)
    print(f"{gain:5.1f} | {al:7.2f} | {ar:7.2f} | {l:.3f}   | {r:.3f}   | {l-r:+.3f}")

# --- Directional test at gain=0.5 with reset ---
print("\n=== Directional test (gain=0.5, with brain reset each case) ===")
fe.AN03A008_GAIN = 0.5
positions = [
    ("top_wall_east",  800,  80, 0.0),     # left eye closer
    ("bot_wall_east",  800, 1120, 0.0),    # right eye closer
    ("top_wall_west",  800,  80, np.pi),   # heading reversed
    ("center_east",    800,  600, 0.0),    # symmetric
    ("left_wall_north",80,   600, 3*np.pi/2),  # near left wall heading north
]
print("pos                | DNa02_L | DNa02_R | diff   | AN03A_L | AN03A_R")
for desc, x, y, heading in positions:
    l, r, al, ar = run_frame(x, y, heading)
    correct = "??"
    if desc == "top_wall_east" and l > r:  correct = "✓"
    elif desc == "bot_wall_east" and r > l: correct = "✓"
    elif desc == "center_east" and abs(l-r) < 0.5: correct = "✓"
    print(f"{desc:22s} | {l:.3f}   | {r:.3f}   | {l-r:+.3f}  | {al:.1f}   | {ar:.1f}  {correct}")

# --- Correlation test: 30 positions along top wall ---
print("\n=== Correlation: r(AN03A008_drive_diff, DNa02_diff) at gain=0.5 ===")
fe.AN03A008_GAIN = 0.5
drive_diffs, dna02_diffs = [], []
for xi in range(100, 1500, 50):  # 28 positions
    l, r, al, ar = run_frame(xi, 80, 0.0)
    drive_diffs.append(al - ar)
    dna02_diffs.append(l - r)

if len(set(dna02_diffs)) > 1:
    rval, pval = stats.pearsonr(drive_diffs, dna02_diffs)
    print(f"r = {rval:.4f}  p = {pval:.4f}")
    print(f"drive_diff: min={min(drive_diffs):.1f} max={max(drive_diffs):.1f}")
    print(f"dna02_diff: min={min(dna02_diffs):.3f} max={max(dna02_diffs):.3f}")
else:
    print("DNa02_diff constant — increase gain or check DN connection")
    print(f"drive_diffs range: {min(drive_diffs):.2f} to {max(drive_diffs):.2f}")
