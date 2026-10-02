"""
FLOW_GAIN reduction test: find threshold where DNa02 is not pre-saturated,
then check if AN03A008 injection provides reliable directional signal.
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

def run_frame(x, y, heading, n_warmup=3, n_avg=10):
    brain.reset_state()
    vx = SPEED * np.cos(heading); vy = SPEED * np.sin(heading)
    drive = encoder.encode(x=x, y=y, vx=vx, vy=vy, heading=heading,
                           world_width=W, world_height=H, margin=40.0)
    brain.drive[:] = drive
    for _ in range(n_warmup): brain.advance(50)
    ls, rs = [], []
    for _ in range(n_avg):
        brain.drive[:] = drive; brain.advance(50)
        l, r = read_dn_rates(brain.counts, groups, 50)
        ls.append(l); rs.append(r)
    return np.mean(ls), np.mean(rs), float(drive[3305]), float(drive[2937])

# -- Part 1: find sub-saturation FLOW_GAIN with AN03A008_GAIN=0 --
fe.AN03A008_GAIN = 0.0
print("\n=== FLOW_GAIN sweep (no AN03A008, center pos, heading east) ===")
print("FLOW_GAIN | DNa02_L | DNa02_R | diff  | avg")
for fg in [5, 10, 20, 30, 40, 50, 75, 100, 150]:
    fe.FLOW_GAIN = fg
    l, r, _, _ = run_frame(800, 600, 0.0)
    avg = (l + r) / 2
    print(f"{fg:9d} | {l:.3f}   | {r:.3f}   | {l-r:+.3f} | {avg:.3f}")

# -- Part 2: at best sub-saturation gain, add AN03A008 and check asymmetry --
# Try FLOW_GAIN=30 and FLOW_GAIN=40
for fg in [30, 40]:
    fe.FLOW_GAIN = fg
    fe.AN03A008_GAIN = 2.0
    print(f"\n=== FLOW_GAIN={fg}, AN03A008_GAIN=2.0: directional test ===")
    print("pos                | AN03A_L | AN03A_R | DNa02_L | DNa02_R | diff")
    tests = [
        ("top_wall_east",  800,  80, 0.0),
        ("bot_wall_east",  800, 1120, 0.0),
        ("center_east",    800,  600, 0.0),
    ]
    for desc, x, y, heading in tests:
        l, r, al, ar = run_frame(x, y, heading)
        print(f"{desc:20s} | {al:7.2f} | {ar:7.2f} | {l:.3f}   | {r:.3f}   | {l-r:+.3f}")

    # Correlation along top wall
    drive_diffs, dna02_diffs = [], []
    for xi in np.linspace(200, 1400, 15):
        l, r, al, ar = run_frame(int(xi), 80, 0.0)
        drive_diffs.append(al - ar)
        dna02_diffs.append(l - r)
    if len(set(dna02_diffs)) > 1:
        rval, pval = stats.pearsonr(drive_diffs, dna02_diffs)
        print(f"Correlation: r={rval:.4f}  p={pval:.4f}  (n={len(dna02_diffs)})")
    else:
        mn = np.mean(dna02_diffs)
        print(f"All zeros — DNa02_diff constant at {mn}")
