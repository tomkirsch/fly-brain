"""
AN03A008 CUDA gain calibration test.
Tests DNa02 L/R differential at realistic speed near walls.
Uses CudaBrain engine (same as production sim).
"""
import sys
from pathlib import Path
import numpy as np
from scipy import stats

DOOMFLY    = Path("/home/mrmotts/fly-brain/doomfly")
FLYBRAIN   = Path("/home/mrmotts/fly-brain/flybrain-sim-repo/flybrain-sim")
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
print(f"  CUDA engine ready ({brain.device_bytes()/1e6:.0f} MB VRAM)")

groups = load_groups(GROUPS_PATH, brain_cpu.ids)
encoder = FlowEncoder(groups, brain_cpu.n)

def run_single(x, y, heading, speed=SPEED):
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    drive = encoder.encode(
        x=x, y=y, vx=vx, vy=vy, heading=heading,
        world_width=W, world_height=H, margin=40.0,
    )
    brain.drive[:] = drive   # CudaBrain host mirror → uploads to GPU in advance()
    brain.advance(STEPS)
    l, r = read_dn_rates(brain.counts, groups, STEPS)  # CudaBrain.counts read back from GPU
    al = float(drive[3305])  # AN03A008_L
    ar = float(drive[2937])  # AN03A008_R
    return l, r, al, ar

print("\n=== CUDA: AN03A008_GAIN sweep at asymmetric position ===")
print("(near top wall y=80, heading east, speed=200)")
print("GAIN  | AN03A008_L | AN03A008_R | DNa02_L | DNa02_R | diff")

for gain in [0.5, 1.0, 2.0, 5.0, 10.0]:
    fe.AN03A008_GAIN = gain
    # Run 5 times and average (CUDA has stochastic variation)
    ls, rs = [], []
    for _ in range(5):
        l, r, al, ar = run_single(800, 80, 0.0)
        ls.append(l); rs.append(r)
    l_mean = np.mean(ls); r_mean = np.mean(rs)
    diff = l_mean - r_mean
    print(f"{gain:5.1f} | {al:8.3f}   | {ar:8.3f}   | {l_mean:.3f}   | {r_mean:.3f}   | {diff:+.3f}")

print("\n=== CUDA: Directional test at gain=2.0 ===")
fe.AN03A008_GAIN = 2.0
positions = [
    ("top_wall_east",  800,  80, 0.0),
    ("bot_wall_east",  800, 1120, 0.0),
    ("top_wall_west",  800,  80, np.pi),
    ("bot_wall_west",  800, 1120, np.pi),
    ("center_east",    800,  600, 0.0),
]
print("pos                | DNa02_L | DNa02_R | diff   | AN03A_L | AN03A_R")
for desc, x, y, heading in positions:
    ls, rs = [], []
    for _ in range(5):
        l, r, al, ar = run_single(x, y, heading)
        ls.append(l); rs.append(r)
    l, r = np.mean(ls), np.mean(rs)
    print(f"{desc:20s} | {l:.3f}   | {r:.3f}   | {l-r:+.3f}  | {al:.2f}   | {ar:.2f}")

print("\n=== CUDA: r(AN03A008_L_drive, DNa02_diff) over 50 frames ===")
fe.AN03A008_GAIN = 2.0
# Run near top wall, vary x position to vary AN03A008_L/R ratio
frames_al, frames_diff = [], []
for xi in range(200, 1400, 24):
    l, r, al, ar = run_single(xi, 80, 0.0)
    frames_al.append(al - ar)
    frames_diff.append(l - r)

if len(set(frames_diff)) > 1 and len(set(frames_al)) > 1:
    r_val, _ = stats.pearsonr(frames_al, frames_diff)
    print(f"r(AN03A008_drive_diff, DNa02_diff) = {r_val:.4f}")
else:
    print("DNa02 diff constant — no correlation possible")
    print(f"drive_diff range: {min(frames_al):.2f} to {max(frames_al):.2f}")
    print(f"DNa02_diff range: {min(frames_diff):.2f} to {max(frames_diff):.2f}")
