# flybrain-sim

Real fly brain (MaleCNS connectome, 166,700 neurons) navigating a 2D world on screen.

Fork of DOOMFLY with the input swapped from a Doom game to synthetic optical flow
and the output swapped from game controls to a WebSocket → browser canvas.

## Prerequisites

- **Python 3.11** (DOOMFLY requires it — `py -3.11` on Windows, `python3.11` on Mac/Linux)
- **NVIDIA GPU** (RTX 3080 class or better) for real-time speed
- **C++ compiler** — `clang++` on Linux/Mac; **not available natively on Windows** (use WSL — see below)

---

## Setup: Windows (via WSL2) — recommended

`build_kernel.py` requires `clang++` and outputs a `.so` file — neither works in native Windows.
Use WSL2 (Ubuntu). The browser renderer `fly.html` opens normally in Windows Chrome/Firefox.

### 1. Open WSL and install deps

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev clang build-essential
```

### 2. Clone DOOMFLY inside WSL (keep native — fast I/O)

```bash
cd ~
mkdir fly-brain && cd fly-brain
git clone https://github.com/nftechie/doomfly doomfly
cd doomfly
```

### 3. Set up venv and install deps

The `--build-constraint` flag in DOOMFLY's README is not supported by older pip. Use
`--no-build-isolation` instead, with `wheel` pre-installed so `brian2` can build:

```bash
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
pip install --upgrade pip
pip install setuptools==68.2.2 numpy==1.24.4 Cython==0.29.37 wheel
pip install -r requirements-neural.txt -r doom/requirements.txt --no-build-isolation
```

### 4. Download MaleCNS data (~1-2 GB)

DOOMFLY's README uses a shell heredoc — use the included cross-platform script instead.
If you ran `download_data.py` on Windows already, copy the files rather than re-downloading:

```bash
# Option A: already downloaded on Windows — copy it over
mkdir -p connectome_data/malecns_v1
cp -r /mnt/c/tom/Projects/fly-brain/fly-brain/doomfly/connectome_data/malecns_v1/* \
      connectome_data/malecns_v1/

# Option B: fresh download
cd ~/fly-brain
python flybrain-sim-repo/flybrain-sim/download_data.py --doomfly ~/fly-brain/doomfly
cd doomfly
```

> **Adjust the `/mnt/c/tom/...` path** to match where your Windows project lives.
> Your `C:\` drive is at `/mnt/c/` in WSL.

### 5. Import graph and compile kernel

```bash
cd ~/fly-brain/doomfly   # make sure you're here
python -m doom.connectome malecns_v1
python -m doom.prepare
python -m doom.build_kernel
```

### 6. Clone flybrain-sim and install its deps

```bash
cd ~/fly-brain
git clone https://github.com/tomkirsch/fly-brain.git flybrain-sim-repo
pip install -r flybrain-sim-repo/flybrain-sim/requirements.txt
pip install pandas pyarrow   # needed for reading DOOMFLY's .feather annotation files
```

### 7. Map neuron groups (one-time)

Always pass `--doomfly` explicitly — the default resolves relative to your working directory,
not the script location, and will fail if you're not in the right folder:

```bash
python flybrain-sim-repo/flybrain-sim/identify_neurons.py --doomfly ~/fly-brain/doomfly
# Saves: flybrain-sim-repo/flybrain-sim/neuron_groups.json
# Expected output: Total groups: 26, empty: 0
```

---

## Setup: Mac/Linux (native)

```bash
cd fly-brain
git clone https://github.com/nftechie/doomfly doomfly
cd doomfly
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
pip install --upgrade pip
pip install setuptools==68.2.2 numpy==1.24.4 Cython==0.29.37 wheel
pip install -r requirements-neural.txt -r doom/requirements.txt --no-build-isolation
python flybrain-sim/download_data.py
python -m doom.connectome malecns_v1
python -m doom.prepare
python -m doom.build_kernel
cd ../flybrain-sim
pip install -r requirements.txt
pip install pandas pyarrow
python identify_neurons.py --doomfly ../doomfly
```

---

## Known gotchas (all platforms)

| Symptom | Fix |
|---|---|
| `source: not found` | Windows — use `.venv-neural\Scripts\Activate.ps1` (PowerShell) |
| `no such option: --build-constraint` | Use `--no-build-isolation` after installing `wheel` (see step 3) |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Missing pinned setuptools — add `setuptools==68.2.2` before main install |
| `error: invalid command 'bdist_wheel'` | `pip install wheel` then retry with `--no-build-isolation` |
| `hashlib has no attribute 'file_digest'` | Python 3.10 — `download_data.py` handles this; make sure you have the latest version |
| `FileNotFoundError: clang++` | Windows native — use WSL2 instead |
| `DOOMFLY not found at .../doomfly` | Pass `--doomfly ~/fly-brain/doomfly` explicitly; don't rely on the default |
| `No annotation CSV found` | DOOMFLY uses `.feather` files — needs `pip install pandas pyarrow` |

---

## Run

Always run from the `flybrain-sim/` directory with `--doomfly` pointing at the DOOMFLY clone:

```bash
cd ~/fly-brain/flybrain-sim-repo/flybrain-sim

# Calibrate first (optional but recommended)
python brain_runner.py --doomfly ~/fly-brain/doomfly --calibrate

# Main loop
python brain_runner.py --doomfly ~/fly-brain/doomfly
```

Open `fly.html` in your browser — double-click it directly (file:// works, no server needed).
On WSL, use the Windows path: `C:\...\flybrain-sim-repo\flybrain-sim\fly.html`.
WSL2 shares localhost with Windows so the WebSocket connection works automatically.

## Calibration

After `--calibrate`, check the printout:
```
dna02_left:  X.X spikes/frame
dna02_right: X.X spikes/frame
Target: dna02_left ~26, dna02_right ~2
```

If off, edit `FLOW_GAIN` in `flow_encoder.py` and rerun.

## File map

```
download_data.py     cross-platform MaleCNS data downloader (replaces DOOMFLY's heredoc)
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
