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

**Confirmed calibration (2026-09-14, RTX 2060, CUDA, `--steps 50`):**

| Constant | Value | Source |
|---|---|---|
| `FLOW_GAIN` | 150.0 | dna02 = 5–6 spikes/window, nactive ~25k |
| `LOOM_GAIN` | 1.0 | lplc2 adj 0.8–1.1 near walls; baseline ~2.0 |
| `BRAIN_LOOM_BASELINE` | 4.0 | DNp01 open-field baseline ~4-5 spikes/window (was 2.0 for LPLC2) |
| `BRAIN_LOOM_THRESHOLD` | 0.4 | DNp01 quantizes 0→4→8; escape fires when adj=4 (esc=8) |
| `BRAIN_LOOM_TURN` | 0.75 | DNp01 adj peak ~4 vs LPLC2's ~1; scaled ÷4 from 3.0 |
| `BRAIN_LOOM_SCATTER` | 8.0 | random kick magnitude; fires when both eyes at esc=8 |
| `TURN_GAIN` | 0.75 | ~3 rad/s at 9x rt; slide up toward 1.5 if rt improves |

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

This distinction is the central project rule: the connectome determines the
spikes *after* we inject synthetic sensory drive, but the surrounding adapter
code determines what those spikes mean in the 2D world.

### Controlled by us

- The synthetic world: arena dimensions, wall geometry, obstacles, initial
  state, collision/clamping, and the browser rendering.
- The synthetic sensors: converting world velocity/wall distance into optical
  flow and looming values, then injecting current directly into annotated
  T4/T5 and LC4/LPLC2 groups. This bypasses the real retina/lamina pipeline.
- The simulation operating point: `FLOW_GAIN`, `LOOM_GAIN`, `--steps`, frame
  timing, baseline subtraction, thresholds, and the single-cell rate
  normalization.
- The motor adapter: `SPEED_GAIN`, `DRAG`, and `TURN_GAIN` convert spike counts
  into pixels/sec and radians/sec. `TURN_GAIN` is not a learned or biological
  constant; it is a user-adjustable unit conversion.
- The escape adapter: DNp01 is the command DN; we read its bilateral differential
  and map to heading change (turn) or a random kick (scatter when both eyes at max).
  DNp01 is a single neuron per side and quantizes coarsely — wall-gradient steering
  is sparse compared to using the LPLC2 population directly.
- **Scripted tactile fallbacks:** `corner_press` (both walls, 4 frames) and
  `wall_press` (single wall, 6 frames) in `world.py` apply heading kicks with no
  neural involvement. These stand in for mechanosensory contact pathways (Johnston's
  organ, leg mechanoreceptors) that are in MaleCNS but not yet wired as inputs.
- The browser interpolation/dead-reckoning and all visual colors/trails.

### Determined by the neural system

- LIF membrane integration, refractory behavior, synaptic delays, synaptic
  weights, recurrent activity, and spike propagation through the 166,700-neuron
  MaleCNS graph.
- The response of downstream neurons to the injected drive, including the
  observed DNa02 and LPLC2 spike counts. We do not manually set those counts.
- Any asymmetry, delay, adaptation, or stochastic variation produced inside
  the connectome. In particular, DNa02 is only one identified neuron per side,
  so 4-vs-8 and occasional zero-count windows are expected single-cell
  observations, not population averages.

### Current honesty boundary

The most neural part of the loop is:

```text
our synthetic flow → real connectome → real DNa02 spikes
```

The least neural part is:

```text
LPLC2 sensor spikes → our direct heading equation → world motion
```

At `TURN_GAIN=0`, DNa02 still fires but it contributes no heading change;
the fly can continue roaming because the separate synthetic-looming adapter
and its programmed corner scatter remain active. Therefore a good-looking
trajectory at zero turn gain does not prove that DNa02 is driving navigation.
Use the diagnostic turn terms below to separate the contributions.

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
- The output mapping (LPLC2 spike differential → escape heading change)

**Now computed from visual geometry (no longer scripted):**
- LC4/LPLC2 drive: ray-cast expansion `v_radial = speed·cos(heading−φ)` summed per
  hemifield. cos(heading−φ) is maximum head-on; complements T4a's sin(heading−φ) which
  is blind head-on. Together they cover all approach angles with no scripted formulas.

**Escape circuit:** We read DNp01, a biologically validated descending neuron (Ache
et al. 2019 Nat Neurosci) that is the top LPLC2 downstream target by aggregate synaptic
weight. It is a genuine command neuron, not a sensor readout. This is a better boundary
than reading LPLC2 directly. GF (Giant Fiber) is absent from the MaleCNS annotation;
DNp01 is the closest analog available in this connectome.

**Baseline subtraction:** DNp01 fires at ~4-5 normalized spikes/window in open field.
`world.py` subtracts `BRAIN_LOOM_BASELINE = 4.0` before threshold and gain, leaving only
the wall-proximity signal above noise.

**Turn diagnostics:** recent `cuda-kernel` builds print and broadcast three
heading-delta components:

```text
turn_dn       DNa02 differential × TURN_GAIN × dt
turn_loom     baseline-subtracted LPLC2 differential × BRAIN_LOOM_TURN × dt
scatter       explicit random corner-breaking kick
```

These are translations applied by `world.py`, not additional neural outputs.
Run with `TURN_GAIN=0` to measure looming-only behavior, then compare with the
same run at a nonzero gain. A stronger test is to temporarily disable looming
and scatter in a controlled experiment, rather than tuning `TURN_GAIN` until
the trajectory looks right.

## Neuron group sizes (MaleCNS)

All descending neurons (DNa02, DNg100, DNg13) are **1 neuron per side** — biologically
correct, these are unique identified cells. DNg100 was found to not be downstream of T4a
in the MaleCNS circuit and was dropped from `read_dn_rates`.

| Group | Count/side | Status |
|-------|-----------|--------|
| T4a/b/c/d | ~835-895 | Input layer, driving well |
| T5a/b/c/d | ~808-863 | Input layer, driving well |
| LC4 | 55-71 | Looming — active, read for nav (baseline ~0.9) |
| LPLC2 | 91-94 | Looming sensor — input to escape circuit; still injected and reported |
| DNa02 | 1 | **Turn signal** — primary motor output |
| DNg100 | 1 | Silent (not T4a downstream) — dropped |
| DNp01 | 1 | **Escape command** — biologically validated (Ache et al. 2019); top LPLC2 downstream target by aggregate synaptic weight; baseline ~4-5 |
| DNp103 | 1 | Fallback escape — highest individual LPLC2→DN synapse weight (R=788); less characterized |

## What to expect

- Fly moves under genuine brain control; DNa02 left/right differential drives turning
- Turns emerge naturally from optical flow asymmetry when heading changes
- Escape driven by DNp01 differential: one eye hits 8 spikes/window, other stays at 4 → heading turn; both at 8 → random scatter kick
- DNp01 is a single neuron per side — quantizes to 0/4/8 only; escape is sparse and command-like (biologically correct)
- Scripted tactile fallbacks: corner-press (4 frames both walls) and wall-press (6 frames single wall) turn the fly away; these are not neural
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

### Recently completed

- **Ray-cast expansion → LPLC2** ✓
  `flow_encoder.py` now computes `v_radial = speed·cos(heading−φ)` per ray and sums
  per hemifield. Removes the last scripted visual input. Confirmed calibration:
  LOOM_GAIN=1.0, lplc2 adj 0.8–1.1 near walls. Head-on escape improved.

- **Find the Giant Fiber / escape command DN → DNp01** ✓
  GF is absent from MaleCNS annotation (only GFC1-4 present). Used `identify_neurons.py
  --trace-from LPLC2` to traverse the connectome and rank downstream partners by
  aggregate synaptic weight. Top result: DNp01 (Ache et al. 2019), biologically
  validated looming-escape command neuron. Wired as primary escape DN; DNp103 is the
  fallback. BRAIN_LOOM_BASELINE recalibrated to 4.0 for DNp01's higher open-field
  baseline.

### Near-term (concrete)

- **Add CLI ablation flags** — `--no-looming` and `--no-scatter` so DNa02-only and
  looming-only experiments are reproducible without editing source. Currently you must
  zero constants manually.

- **Mechanosensory injection** — replace scripted corner-press/wall-press with genuine
  neural input. Real flies use Johnston's organ (antennal) and leg mechanoreceptors for
  contact/collision detection; these neurons exist in MaleCNS. Steps: (1) search MaleCNS
  annotation for JON/chordotonal neuron types, (2) inject contact-triggered current when
  fly is at wall boundary, (3) calibrate and verify downstream escape circuit fires.
  Moderate effort (~same as adding LC4/LPLC2 expansion signal). Main unknown: whether
  the annotated mechanosensory types in MaleCNS actually connect to escape circuits.

- **Add obstacles** — `world.obstacles` accepts `{cx, cy, r}` dicts already; just populate
  them. Tests richer navigation and whether expansion signal handles convex obstacles the
  same way it handles walls.

- **Adaptive BRAIN_LOOM_BASELINE** — currently hardcoded at 2.0. Could compute a 200-frame
  rolling mean of lplc2 output in open field and subtract that dynamically. Would survive
  FLOW_GAIN or LOOM_GAIN changes without manual recalibration.

### Medium-term

- **More motor DNs** — only DNa02 drives turning and speed. Drosophila has ~50+ descending
  neurons per side. Run `identify_neurons.py` with a broader query (DNg*, DNc*, etc.) and
  see which ones actually fire above baseline during navigation. Could find a speed-specific
  DN separate from the turn DN.

- **Decouple physics from brain loop** — run `world.step()` at 60fps using last-known DN
  rates; update DN rates async when each brain frame completes. Makes on-screen motion
  smooth even at rt=9x. Requires a thread-safe rate buffer.

- **Brain spike visualizer** — pipe per-neuron counts over WebSocket → Three.js / canvas
  point cloud. Even a 2D projection of the optic lobe showing which columns are hot would
  be impressive and useful for debugging.

### Longer-term / speculative

- **Central complex (EPG compass neurons)** — the ring attractor encoding heading is in
  MaleCNS. If we inject visual landmark cues into the appropriate columnar neurons (E-PG,
  P-EN) and read the population vector, we could replace the programmed `self.heading`
  variable with a brain-computed heading estimate. This would be the most significant
  neural-ness upgrade possible without adding real optics.

- **Haltere-like proprioception** — real flies stabilize flight using halteres (gyroscopes).
  Could inject angular-velocity signals into campaniform sensillum homologs each frame,
  closing the sensorimotor loop for heading stabilization.

- **Performance** — CUDA runs at ~9x realtime on RTX 2060 for 166k neurons. Profiling the
  kernel may reveal memory-bandwidth bottlenecks. Sparse-update strategies (only propagate
  from neurons that actually spiked) could help; the CPU engine already does this via the
  active set, but the CUDA port runs all neurons every tick.
