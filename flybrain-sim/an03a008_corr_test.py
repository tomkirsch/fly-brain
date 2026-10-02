"""Final correlation test at AN03A008_GAIN=2.0 with many positions and resets."""
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

def run_frame(x, y, heading, n_warmup=3, n_avg=8):
    brain.reset_state()
    vx = SPEED * np.cos(heading); vy = SPEED * np.sin(heading)
    drive = encoder.encode(x=x, y=y, vx=vx, vy=vy, heading=heading,
                           world_width=W, world_height=H, margin=40.0)
    brain.drive[:] = drive
    for _ in range(n_warmup):
        brain.drive[:] = drive; brain.advance(50)
    ls, rs = [], []
    for _ in range(n_avg):
        brain.drive[:] = drive; brain.advance(50)
        l, r = read_dn_rates(brain.counts, groups, 50)
        ls.append(l); rs.append(r)
    return np.mean(ls), np.mean(rs), float(drive[3305]), float(drive[2937])

fe.AN03A008_GAIN = 2.0

# Test 1: positions along top wall varying x (should vary AN03A008_L/R with fixed heading east)
print("\nTop wall (y=80), heading east, x varies (should show L>R consistently):")
drive_diffs, dna02_diffs = [], []
for xi in np.linspace(200, 1400, 20):
    l, r, al, ar = run_frame(int(xi), 80, 0.0)
    drive_diffs.append(al - ar)
    dna02_diffs.append(l - r)
    print(f"  x={xi:4.0f} AN03A_diff={al-ar:+6.1f} DNa02_diff={l-r:+.3f}")

if len(set(dna02_diffs)) > 1:
    r_val, p_val = stats.pearsonr(drive_diffs, dna02_diffs)
    print(f"r(drive_diff, dna02_diff) = {r_val:.4f}  p = {p_val:.4f}")
else:
    print("constant -- no variance")

# Test 2: compare top wall vs bottom wall (should flip sign)
print("\nTop wall vs bottom wall (x=800, heading east):")
l1, r1, al1, ar1 = run_frame(800, 80, 0.0)
l2, r2, al2, ar2 = run_frame(800, 1120, 0.0)
print(f"  top_wall: AN03A_L={al1:.1f} AN03A_R={ar1:.1f} DNa02_L={l1:.3f} DNa02_R={r1:.3f} diff={l1-r1:+.3f}")
print(f"  bot_wall: AN03A_L={al2:.1f} AN03A_R={ar2:.1f} DNa02_L={l2:.3f} DNa02_R={r2:.3f} diff={l2-r2:+.3f}")
correct = "✓ signs flipped" if (l1-r1) * (l2-r2) < 0 else "✗ same sign (wrong)"
print(f"  {correct}")
