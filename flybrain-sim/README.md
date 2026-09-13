# flybrain-sim

Real fly brain (MaleCNS connectome, 166,700 neurons) navigating a 2D world on screen.

Fork of DOOMFLY with the input swapped from a Doom game to synthetic optical flow
and the output swapped from game controls to a WebSocket → browser canvas.

## Prerequisites

Needs Python 3.11 and an NVIDIA GPU (RTX 3080 class or better) for real-time speed.

### 1. Clone DOOMFLY

**Mac/Linux:**
```bash
cd fly-brain
git clone https://github.com/nftechie/doomfly doomfly
cd doomfly
```

**Windows:**
```cmd
cd fly-brain
git clone https://github.com/nftechie/doomfly doomfly
cd doomfly
```

### 2. Set up DOOMFLY

**Mac/Linux:**
```bash
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
pip install setuptools==68.2.2 numpy==1.24.4 Cython==0.29.37
pip install -r requirements-neural.txt -r doom/requirements.txt
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv-neural
.venv-neural\Scripts\Activate.ps1
pip install setuptools==68.2.2 numpy==1.24.4 Cython==0.29.37
pip install -r requirements-neural.txt -r doom/requirements.txt
```

> Note: DOOMFLY's README uses `--build-constraint` which older pip versions don't support.
> The two-step install above achieves the same result.

Then download the MaleCNS data (~1-2 GB). DOOMFLY's README uses a shell heredoc that
doesn't work on Windows — use the download script instead:

```
cd ..
python flybrain-sim/download_data.py
cd doomfly
```

Then import and compile (same on all platforms):
```
python -m doom.connectome malecns_v1
python -m doom.prepare
python -m doom.build_kernel
```

### 3. Install our deps (in the same venv)

**Mac/Linux:**
```bash
cd ../flybrain-sim
pip install -r requirements.txt
```

**Windows:**
```cmd
cd ..\flybrain-sim
pip install -r requirements.txt
```

### 4. Map neuron groups (one-time)
```
python identify_neurons.py --doomfly ../doomfly
# Creates neuron_groups.json
```

## Run

```
# Optional: calibrate flow gain first
python brain_runner.py --calibrate

# Main loop
python brain_runner.py
```

Open `fly.html` in a browser (file:// works, no server needed).

## Calibration

After first run with `--calibrate`, check the printout:
```
dna02_left:  X.X spikes/frame
dna02_right: X.X spikes/frame
Target: dna02_left ~26, dna02_right ~2
```

If off, edit `FLOW_GAIN` in `flow_encoder.py` and rerun.

## File map

```
identify_neurons.py  one-time: maps T4/T5/DN neuron IDs → neuron_groups.json
flow_encoder.py      velocity + heading → T4/T5/LC4 drive currents
world.py             2D physics: position, walls, looming sensor
ws_server.py         asyncio WebSocket broadcaster (background thread)
brain_runner.py      main loop: encode → step brain → read DN → update world → broadcast
fly.html             browser canvas renderer (open directly, no server)
```

## Architecture

```
  world state (pos, velocity, heading)
        ↓
  flow_encoder → brain.drive[T4/T5/LC4 indices]
        ↓
  Brain.advance() × 200 mini-ticks
        ↓
  brain.counts[DNa02/DNg100 indices] → left/right rate
        ↓
  world.step(left, right) → update pos/heading/looming
        ↓
  ws_server.broadcast() → fly.html (canvas)
```

## What to expect

- Fly wanders using the brain's own optomotor and escape reflexes
- Approaching walls → looming neurons fire → Giant Fiber → escape turn
- Visual flow stabilization kicks in automatically
- Wings flap in browser based on speed; red glow = looming active

## Next

- Add obstacles (circles) to world.obstacles for richer navigation
- Add brain viz: pipe spike counts per neuron over WebSocket → Three.js point cloud
- Tune SPEED_GAIN / TURN_GAIN in world.py to taste
