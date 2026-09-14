# flybrain-sim

Real fly brain (MaleCNS connectome, 166,700 neurons) navigating a 2D world on screen.

Fork of DOOMFLY with the input swapped from a Doom game to synthetic optical flow
and the output swapped from game controls to a WebSocket → browser canvas.

## GPU acceleration (CUDA)

`cuda-kernel` branch adds `engine_cuda.py`: the same LIF kernel ported to a
numba `cuda.jit` path that runs **all 166,700 neurons every tick** (no active-set
pruning needed on GPU).

### WSL2 setup (RTX 2060 6GB is sufficient — ~450 MB VRAM used)

Install the CUDA toolkit inside WSL (the Windows driver already provides the
user-mode driver; WSL needs the toolkit only for numba's driver binding):

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit        # or: conda install -c nvidia cuda-toolkit
python -c "import numba.cuda; print(numba.cuda.is_available())"   # expect True
```

Then run normally:

```bash
python brain_runner.py                # auto-detects GPU, uses CUDA engine
python brain_runner.py --cpu          # force the CPU engine
python brain_runner.py --selftest     # 200 steps CPU vs GPU, prints mismatch
```

Auto-detection: if `numba.cuda.is_available()` is False (or `--cpu` is passed),
brain_runner prints a warning and uses the original CPU engine unchanged.

Selftest runs 200 steps on both engines from identical initial state and drive
seed, then prints per-neuron spike-count mismatch (max / mean). Zero or a
handful of ±1 mismatches is expected (float32 delivery-order rounding); large
mismatch means a real port bug.

### Version compatibility notes (WSL2, driver 610.43, CUDA UMD 13.3)

- numba must be new enough to bind a CUDA 13 user-mode driver: use
  **numba >= 0.62** (`pip install -U numba`). Older numba (≤0.61) supports only
  CUDA 12.x and will fail with `NvvmError` / `cuInit` errors.
- Keep `numpy < 2.4` if numba complains at import time.
- If `numba.cuda.is_available()` prints False despite a working `nvidia-smi`,
  check `CUDA_HOME` points at the toolkit and try `conda install -c nvidia
  cuda-toolkit` (conda toolkits tend to match numba's supported versions).

---

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

### 8. Enable CUDA for GPU acceleration (WSL2)

The sim defaults to CPU (Numba njit). For real-time speed, get the GPU kernel working.

**Step 1 — do NOT install `nvidia-cuda-toolkit` via apt.** It installs a CUDA 11 stub
`libcuda.so.1` into `/lib/x86_64-linux-gnu/` that shadows the real WSL2 driver and causes
`CUDA_ERROR_NO_DEVICE (100)` even though `nvidia-smi` works fine. If you already installed it:

```bash
sudo apt remove nvidia-cuda-toolkit
sudo ldconfig
```

**Step 2 — force the WSL2 driver** by prepending its path:

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib' >> ~/.bashrc
source ~/.bashrc
python -c "import numba.cuda; print(numba.cuda.is_available())"  # must print True
```

**Step 3 — install the pip CUDA wheels** (provide nvcc, nvrtc, libdevice):

```bash
pip install numba -U   # needs ≥0.62; 0.67 confirmed working
pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 cuda-python
```

**Step 4 — libdevice symlink.** Numba doesn't scan the pip wheel layout for `libdevice.10.bc`
automatically; symlink it to the standard path it always checks:

```bash
sudo mkdir -p /usr/local/cuda/nvvm/libdevice/
sudo ln -sf \
  $(find ~/.venv-neural -name "libdevice.10.bc" 2>/dev/null | head -1) \
  /usr/local/cuda/nvvm/libdevice/libdevice.10.bc
```

(If your venv isn't at `~/.venv-neural`, adjust the find path.)

**Verify:**

```bash
python brain_runner.py --doomfly ~/fly-brain/doomfly --selftest
# Expected: PASS, mismatching neurons <100, max |diff| = 1
```

If the selftest passes, `brain_runner.py` will auto-detect CUDA and run on GPU.
Use `--cpu` to force CPU mode.

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
| Calibration hangs for minutes after "WebSocket:" line | **Do NOT apply DOOMFLY's lamina bias (12.0)** — it ignites all 166k neurons, making CPU advance() take minutes per frame. We inject directly at T4/T5 instead; they're seeded into `brain.active` so drive reaches them. |
| `L=0.0 R=0.0` even after first few frames | T4/T5 neurons not in initial active set — brain never processes them. `_seed_active_set()` in `brain_runner.py` fixes this. |

---

## Run

Always activate the venv first, then run from the `flybrain-sim/` directory:

```bash
source ~/fly-brain/doomfly/.venv-neural/bin/activate  # venv lives in doomfly, not flybrain-sim
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

After `--calibrate`, check the printout for healthy values:
```
Good: dna02 stable at 3-8 spikes/window, nactive < 100k.
```

**On the "26 Hz vs 2 Hz" boat.horse target:** that's a biological Hz figure.
At 200 ticks/window (20ms), 26 Hz = 0.52 spikes/window — already well below
what we see. The sim LIF runs hotter than biology; 3-8 spikes/window is fine.
Do NOT raise FLOW_GAIN trying to hit 26 counts — it's physically impossible
for a single neuron (refractory period caps max at ~9 spikes/200 ticks).

If `dna02 = 0`: lower FLOW_GAIN.
If `nactive > 150k`: lower FLOW_GAIN.

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
  flow_encoder → brain.drive[T4/T5/LC4/LPLC2 indices]
        ↓
  Brain.advance() × N ticks (default 50; --steps to override)
        ↓
  brain.counts[DNa02_left/right] normalized to 200-tick equivalent
        ↓
  world.step(left, right, dt=actual_elapsed) → update pos/heading/looming
        ↓
  ws_server.broadcast() → fly.html (60fps interpolated canvas)
```

## What's neural vs what's programmed

**Genuinely from the 166k-neuron connectome:**
- DNa02 spike magnitude (~5-6 spikes/200-tick window) — real T4a → medulla → lobula → DN chain
- Left/right differential — when lateral velocity creates asymmetric optical flow, the
  MaleCNS circuit routes it asymmetrically to dna02_left vs dna02_right through real
  synaptic weights; the 4/8 alternation is genuine stochastic single-neuron firing
- LC4/LPLC2 looming response — LPLC2 baseline ~2.1 normalized from T4/T5 background
  connectivity; rises to ~3.5 near walls. The circuit responds are real.

**Programmed by us (`world.py`, `flow_encoder.py`):**
- The physics equations (speed/heading update from DN rates)
- SPEED_GAIN, TURN_GAIN, DRAG constants
- Direction convention mapping velocity → T4a subtypes
- The looming input injection itself (geometric wall distance → LPLC2 drive)
- The output mapping (LPLC2 spike differential → escape heading change)

**Architecture note on the looming path:** We're reading LPLC2 (a *sensor* neuron
encoding "looming from the left") and mapping it directly to heading change. A more
biologically honest approach would find a downstream *command* DN in the escape circuit
(Giant Fiber → thoracic) and read that instead. The current approach works but skips
the output stage of the escape circuit.

**Baseline subtraction:** LPLC2 fires at ~2.1 normalized even in open field (T4/T5
bleed-through). To get a clean wall-proximity signal, `world.py` subtracts
`BRAIN_LOOM_BASELINE = 2.0` before threshold and gain. This is analogous to
contrast normalization in a downstream neuron adapting to the background drive.

## Neuron group sizes (MaleCNS)

All descending neurons (DNa02, DNg100, DNg13) are **1 neuron per side** — biologically
correct, these are unique identified cells. DNg100 was found to not be downstream of T4a
in the MaleCNS circuit and was dropped from `read_dn_rates`.

| Group | Count/side | Status |
|-------|-----------|--------|
| T4a/b/c/d | ~835-895 | Input layer, driving well |
| T5a/b/c/d | ~808-863 | Input layer, driving well |
| LC4 | 55-71 | Looming — active, read for nav (baseline ~0.9) |
| LPLC2 | 91-94 | **Looming escape** — wired to navigation (baseline ~2.1) |
| DNa02 | 1 | **Turn signal** — primary motor output |
| DNg100 | 1 | Silent (not T4a downstream) — dropped |

## What to expect

- Fly moves under genuine brain control; DNa02 left/right differential drives turning
- Turns emerge naturally from optical flow asymmetry when heading changes
- Wall avoidance driven by LPLC2 differential: left looms more → escape right
- Looming ring in browser split into left/right hemispheres showing which eye is hotter
- Wings flap in browser based on speed
- Stochastic: DNa02 is a single neuron per side — it will occasionally fire 0 spikes
  in a 50-tick window; that's real single-cell noise, not a bug

## Signal delay with --steps 50

T4a → DNa02 takes ~4 synaptic hops × 18 ticks = 72 ticks minimum. With `--steps 50`,
one window is too short to span the full chain in a single pass. Spikes continue
propagating across advance() calls (spike ring buffer and voltages persist), so DNa02
does fire correctly — but it reflects T4a state from ~3 windows ago. At rt=9x that's
~135ms of real-time lag. Acceptable for navigation; noticeable for fast reactive control.
Use `--steps 200` for full within-window propagation (slower, ~3fps).

## Next

- **Find a command DN downstream of the Giant Fiber escape circuit** in the MaleCNS graph.
  Something that fires when LPLC2 fires and encodes escape direction. Read that instead
  of reading LPLC2 directly — would translate at the *command* level (like DNa02) rather
  than the *sensor* level (like current LPLC2 approach).
- Add obstacles (circles) to `world.obstacles` for richer navigation
- Brain viz: pipe spike counts per neuron over WebSocket → Three.js point cloud
- Decouple physics from brain loop: run physics at 60fps with last-known DN rates,
  update DN rates async when each brain frame completes → smoother on-screen motion
