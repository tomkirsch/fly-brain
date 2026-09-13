# Fly Brain

**Source:** https://boat.horse/fly/
**Ingested:** 2026-09-13

## What It Is

A working real-time simulation of a complete male fruit fly nervous system — all 166,700 neurons and 25.6 million synaptic connections — based on the **MaleCNS connectome** released by HHMI Janelia + Google Research in 2026.

It's not just a visualization. It's a closed-loop system with actual sensors and outputs:

- **Eyes:** A webcam feeds visual input. Optical flow (Farnebäck algorithm) detects motion and looming; signals are injected at the T4/T5 (direction-selective) and LC4/LPLC2 (looming) neurons, bypassing the graded/non-spiking early visual stages that go silent in a spiking model.
- **Brain:** Leaky integrate-and-fire neurons, stepped 200x per 20ms tick (~0.1ms each). Runs ~1.46× real time on GPU.
- **Motor output:** Motor neuron activity drives a **16×16 LED map** of the fly's body (via DDP/WLED) AND the legs of a **Strandbeest** walking machine (servo commands via UDP at 50Hz).
- **Feedback loop:** The Strandbeest's IMU (gyro/accel) and servo speeds feed back into the fly's balance sense (halteres) and leg-position sense (chordotonal organs), closing the loop.

## Key Details

- Neurons run as **leaky integrate-and-fire** units, parameters from Shiu et al. 2024 (Nature)
- Simulation ported from **DOOMFLY** schedule (also see Fly64, flybrain.online)
- The differentiator from other MaleCNS demos: injection point is *past* the dead non-spiking visual stages — this makes downstream responses actually fire
- An antennal-lobe / mushroom-body feedback loop ignites above ~7mV stimulus and keeps the network humming at ~12Hz — emergent persistent activity
- Giant Fiber = escape command neuron (looming → 132Hz); DNa02 handles left/right visual asymmetry (26Hz vs 2Hz)
- Validation: sugar taste → proboscis motor neurons, replicating a known Shiu et al. experiment

## Why It's Interesting

- Full connectome → physical embodiment pipeline, closed loop
- The "injection point" problem is a real solved engineering challenge — knowing where the graded visual stages break down and routing around them
- Mushroom body ignition / self-sustaining network activity is emergent behavior from the real connectome, not engineered in
- The Strandbeest as motor output is a conceptually perfect choice — a mechanically simple multi-leg walker, kinetically analogous to insect locomotion

## Brainstorm Hooks (Seafoam/Art angles)

- **Poofer integration?** Motor neurons → fire control instead of leg servos
- **Crowd as retina?** Camera tracking crowd movement as visual input to a live brain simulation on a display
- **Reactive installation:** Fly brain responding to music/audio (convert to visual flow or inject into auditory-analog neurons), driving light output
- **Scale up / slow down:** Project the LED body map huge; render neural spike activity in real time as an art piece
- **Sound:** Sonify spike rates per neuron group — fly brain as live generative instrument

### Music Response (art piece concept)

**The circuit:** JO-B neurons (Johnston's organ, second antennal segment) are the fly's ears — specifically tuned to ~160-225Hz oscillatory signals, which is the male courtship song frequency. Male flies vibrate one wing to produce a pulse song (~170ms IPI) or sine song (~200Hz continuous). Females use JO-B to evaluate the song and choose whether to mate.

**Wiring audio in:**
```
microphone → FFT → bandpass 160-225Hz → amplitude → scale × GAIN → brain.drive[jоб_indices]
```

**Genre sensitivity:**
- Bass-heavy EDM/hip-hop: strong 60-180Hz content → heavy JO-B activation
- Punk/guitar rock: mostly 200-5000Hz → JO-B gets nothing, fly ignores it
- Cello, bass clarinet, male voice: sits right in the activation band
- Beat drops vs silence: behavioral change with the music, not programmed

**Why it's interesting:** In a real fly, JO-B → courtship downstream circuits → mating behavior. In our sim there's no mate context. Those circuits fire, propagate through the mushroom body, hit descending neurons in some unpredetermined pattern. The fly might slow down, turn erratically, oscillate. Nobody knows in advance — the biology determines the music response, not the programmer.

**The art proposition:** A fly that reacts to live music in ways that are biologically grounded but behaviorally unpredictable. A performer plays in front of it and sees what the connectome does with their sound. No programmer decided what the music means to the fly. Evolution did, 50 million years ago.

## Web/Graphical Embodiment — Build Direction

### The Concept

Instead of a Strandbeest, the fly lives on screen. Motor neurons drive a sprite; the sprite's movement through a textured world generates synthetic optical flow that feeds back into the brain. Fully closed loop, no physical hardware.

Emergent behaviors this should produce for free:
- **Optomotor reflex** — drift correction, the fly stabilizes heading
- **Wall avoidance** — looming → Giant Fiber → escape turn
- **Open-field wandering** — mushroom body buzz keeps it active without stimulus

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python / CUDA (sim process)                        │
│                                                     │
│  World state (fly x,y, heading, obstacles)          │
│       ↓                                             │
│  Synthetic optical flow from velocity               │
│  (left/right eye split, direction vectors)          │
│       ↓                                             │
│  Inject → T4/T5 + looming neurons                  │
│       ↓                                             │
│  Step 200 LIF ticks (0.1ms each)                   │
│       ↓                                             │
│  Read motor neurons → left/right fire rates        │
│       ↓                                             │
│  Update fly velocity + position in world state      │
│       ↓                                             │
│  WebSocket broadcast: {pos, heading, motor_state,  │
│                         neuron_activity (optional)} │
└──────────────────────────┬──────────────────────────┘
                           │ ~50fps
                    ┌──────▼──────┐
                    │   Browser   │
                    │             │
                    │  Canvas /   │
                    │  Three.js   │
                    │             │
                    │  fly sprite │
                    │  world bg   │
                    │  brain viz  │
                    │  (optional) │
                    └─────────────┘
```

**Key principle:** Python owns the world state and the loop. The browser is a pure renderer — it just draws whatever Python tells it. No latency in the feedback path because the loop never leaves Python.

### Path A: Synthetic 2D world (start here)

- Fork DOOMFLY; gut the webcam input and servo output
- Replace input: compute optical flow vectors from fly velocity + heading each frame
- Replace output: WebSocket server broadcasts motor state + world state
- Add virtual obstacles/walls — when fly approaches one, compute looming signal (expanding object) and inject into looming neurons
- Browser: fly sprite on textured canvas, walls, maybe leave a trail

**What you need to build:**
1. `world.py` — fly position, velocity, wall/obstacle map, collision
2. `flow_encoder.py` — velocity + heading → left/right eye optical flow vectors (replaces webcam + Farnebäck)
3. `ws_server.py` — asyncio WebSocket, broadcasts frame data
4. `fly.html` — canvas renderer, WebSocket client

### Path B: 3D brain visualization overlay (add later)

The connectome data includes 3D anatomical coordinates for every neuron. Render them as a point cloud in Three.js alongside the moving fly. Color each point by current firing rate. Watch signal propagate eye → brain → motor in real time.

This is independent of Path A — it's just an additional data stream from the WebSocket.

### Path C: Full 3D world (future)

Render a first-person fisheye view from the fly's POV each frame, feed that as the visual input instead of synthetic flow. The fly genuinely navigates a 3D space using its real visual system. Much more work; behavior difference may not justify it until Path A is proven.

### Implementation Notes (from source investigation)

**Base: fork DOOMFLY, not boat.horse** (boat.horse code isn't public yet). DOOMFLY has the same LIF kernel and MaleCNS graph loading. Three targeted changes get us to Path A.

**Injection mechanism:** DOOMFLY exposes a `brain.drive` float array (one slot per neuron). Set it before each step — the LIF kernel adds it as continuous input current. DOOMFLY sets it for retina neurons. We set it for T4/T5 neurons instead. Identical API, different target indices.

**Finding T4/T5 indices:** MaleCNS annotation CSV has `type` column (`T4a`, `T4b`, `T5a`, etc.) and body side (`L`/`R`). Filter → get biological IDs → `np.searchsorted(brain.ids, bio_ids)` → internal indices. One-time `identify_neurons.py` script, saves `neuron_groups.json`.

**Synthetic flow formula:**
- Forward speed → T4a (front→back) in both eyes, proportional to speed
- Turning right → left eye gets extra T4a (front→back), right eye gets T4b (back→front)
- Looming → LC4 + LPLC2 neurons, proportional to rate of approach to nearest wall

**Calibration target:** boat.horse reports T4a subtype-a in both eyes → DNa02 fires 26Hz (excited side) vs 2Hz (opposite). Use this as validation: inject a fixed flow signal, check if DNa02 fires in that range.

**Motor output:** Read descending neuron (DN) fire rates from `brain.counts` after each step. DNa02/DNg13 = left/right differential → turning. DNg100 = forward. Same neurons Fly64 uses, well-characterized.

**DOOMFLY Numba kernel call** (from engine.py source):
```python
brain.cursor = advance(
    brain.ptr, brain.post, brain.weight,
    brain.v, brain.g, brain.refractory, brain.drive,
    brain.queue, brain.queue_count, brain.cursor,
    200, brain.dt, brain.counts,
    brain.active, brain.active_flag, brain.nactive
)
```

**Fly64 finding:** runs on Mac M2 CPU (no NVIDIA). Slower than real-time but functional. For interactive art use, ~0.2× real time on CPU may be acceptable. GPU gets to 1.46× real time.

### Hardware Note

Needs NVIDIA GPU (RTX 3080 class or better) for real-time. CPU-only works but slower. Browser display can be any machine on the same network.

## References

- MaleCNS connectome: https://www.janelia.org/project-team/flyem/male-cns-connectome
- DOOMFLY: https://github.com/nftechie/doomfly
- Fly64: https://github.com/ornata/fly
- flybrain.online
- Shiu et al. 2024: https://www.nature.com/articles/s41586-024-07763-9
