"""
CUDA port of the DOOMFLY LIF kernel (doom/engine.py `advance`).

Semantics preserved exactly (dt=0.1 ms, 1.8 ms transmission delay, threshold
-45 mV, reset to -52 mV / g=0 / 2.2 ms refractory, delivery-after-integration
schedule, Brian2-style "(unless refractory)" synaptic-write suppression):

  - Per tick, every neuron (ALL 166,700 — no active-set pruning on GPU):
      1. if refractory>0: decrement; when it hits 0 (same tick) integrate.
      2. integrate:  v = -52 + (v+52)*av + drive*(1-av) + g*coupling
                     g *= ag            (av=exp(-dt/20), ag=exp(-dt/5))
      3. threshold:  if v > -45: counts[i]+=1; spike is scheduled into
         delay slot (cursor+18) % 19 and the neuron is reset immediately
         (v=-52, g=0, refractory=22) — identical net effect to the CPU
         kernel's post-delivery reset loop.
  - Delivery: spikes from 18 ticks ago add weight to postsynaptic g,
    skipping refractory postsynaptic neurons.

GPU restructuring (vs the CPU scatter loop):
  - The CPU kernel scatters along each spiker's out-edges. Threads cannot
    scatter cheaply, so we invert the graph once at init into a reverse CSR
    (incoming edges per neuron) and each thread GATHERS its incoming spikes.
  - The spike queue (queue/queue_count) is replaced by a 19-slot flag ring:
    integrate sets spike_ring[f, i]=1 for its delay slot; deliver reads
    spike_ring[s, ·] for s = cursor % 19 and clears it. Same 1.8 ms delay.
  - Delivery accumulation is done in-edge order (deterministic); the CPU
    accumulates in spike-queue order, so float32 rounding can differ in the
    last ULP — the --selftest comparison tolerates this (zero-ish mismatch).

Memory on device (RTX 2060 6GB is sufficient, ~450 MB total):
  post, weight, reverse-CSR (cptr, cin, wrev), v, g, drive, refractory,
  counts, spike_ring (19 × n uint8). v/g/refractory stay resident; per step
  only `drive` is uploaded and `counts` (plus a spike total) is read back.
"""

import math
import time

import numpy as np
from numba import cuda

DELAY_SLOTS = 19          # int(round(1.8/dt)) + 1 for dt=0.1
DELAY_TICKS = 18          # int(round(1.8/dt))
RESET_REFRACTORY = 22     # int(round(2.2/dt))
V_RESET = np.float32(-52.0)
V_THRESH = np.float32(-45.0)

THREADS = 256


def cuda_available() -> bool:
    try:
        import numba.cuda as _c
        return bool(_c.is_available())
    except Exception:
        return False


@cuda.jit
def _integrate_kernel(v, g, refractory, drive, counts, spike_ring,
                      cursor, f, dt):
    av = math.exp(-dt / 20)
    ag = math.exp(-dt / 5)
    coupling = (av - ag) / 3
    i = cuda.grid(1)
    if i >= v.size:
        return
    if refractory[i] > 0:
        refractory[i] -= 1
    if refractory[i] == 0:
        v[i] = -52 + (v[i] + 52) * av + drive[i] * (1 - av) + g[i] * coupling
        g[i] *= ag
        if v[i] > V_THRESH:
            counts[i] += 1
            spike_ring[f, i] = 1
            # Same-tick reset (CPU kernel resets these right after delivery;
            # delivery targeting them is erased either way, net effect equal).
            v[i] = V_RESET
            g[i] = 0
            refractory[i] = RESET_REFRACTORY


@cuda.jit
def _deliver_kernel(cptr, cin, wrev, g, refractory, spike_ring, s):
    j = cuda.grid(1)
    if j >= g.size:
        return
    if refractory[j] > 0:
        return  # Brian2 "(unless refractory)": no synaptic writes
    acc = 0.0
    for e in range(cptr[j], cptr[j + 1]):
        if spike_ring[s, cin[e]] != 0:
            acc += wrev[e]
    if acc != 0.0:
        g[j] += acc


@cuda.jit
def _clear_slot_kernel(spike_ring, s):
    i = cuda.grid(1)
    if i < spike_ring.shape[1]:
        spike_ring[s, i] = 0


class CudaBrain:
    """Wraps a loaded doom.engine.Brain and runs advance() on the GPU.

    Exposes the attributes brain_runner uses: n, ids, ptr, post, weight,
    drive (host mirror), counts (read back per advance), cursor, dt,
    nactive (GPU proxy: total spikes this advance — display only).
    """

    def __init__(self, brain):
        if brain.dt != 0.1:
            raise ValueError('CUDA kernel supports only dt=0.1 ms.')
        self._cpu = brain              # keep host graph + validation intact
        self.n = brain.n
        self.ids = brain.ids
        self.ptr = brain.ptr
        self.post = brain.post
        self.weight = brain.weight
        self.dt = brain.dt
        self.cursor = 0

        # ---- reverse CSR (incoming edges per neuron), built once ----
        t0 = time.perf_counter()
        n = self.n
        in_deg = np.bincount(self.post, minlength=n).astype(np.int64)
        cptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(in_deg, out=cptr[1:])
        order = np.argsort(self.post, kind='stable')        # edge idx by target
        cin = np.repeat(np.arange(n, dtype=np.int32), in_deg)  # source neuron
        cin = cin[order].astype(np.int32)
        wrev = self.weight[order].astype(np.float32)
        print(f"  reverse CSR built in {time.perf_counter()-t0:.1f}s "
              f"({len(order):,} in-edges)")

        # ---- device allocation (once) ----
        mem = self._d = {}
        mem['post'] = cuda.to_device(self.post)
        mem['weight'] = cuda.to_device(self.weight)
        mem['cptr'] = cuda.to_device(cptr)
        mem['cin'] = cuda.to_device(cin)
        mem['wrev'] = cuda.to_device(wrev)
        mem['v'] = cuda.to_device(np.full(n, -52, dtype=np.float32))
        mem['g'] = cuda.to_device(np.zeros(n, dtype=np.float32))
        mem['drive'] = cuda.to_device(np.zeros(n, dtype=np.float32))
        mem['refractory'] = cuda.to_device(np.zeros(n, dtype=np.int16))
        mem['counts'] = cuda.to_device(np.zeros(n, dtype=np.int32))
        mem['spike_ring'] = cuda.to_device(
            np.zeros((DELAY_SLOTS, n), dtype=np.uint8))
        self.blocks = (n + THREADS - 1) // THREADS

        # host mirrors
        self.drive = np.zeros(n, dtype=np.float32)
        self.counts = np.zeros(n, dtype=np.int32)
        self.nactive = np.zeros(1, dtype=np.int32)
        self.total_spikes = 0
        self.sim_ms = 0.0
        self._zeros_counts = np.zeros(n, dtype=np.int32)  # reused each advance

    def reset_state(self, v=-52.0):
        """Reset v/g/refractory/counts on device (init + calibration)."""
        n = self.n
        self._d['v'].copy_to_device(np.full(n, v, dtype=np.float32))
        self._d['g'].copy_to_device(np.zeros(n, dtype=np.float32))
        self._d['refractory'].copy_to_device(np.zeros(n, dtype=np.int16))
        self.reset_counts()

    def reset_counts(self):
        self._d['counts'].copy_to_device(np.zeros(self.n, dtype=np.int32))
        self.counts[:] = 0

    def advance(self, steps: int) -> int:
        d = self._d
        t0 = time.perf_counter()
        # Reset device counts each advance window so brain_runner reads per-window spikes,
        # not a running total. host-side brain.counts[:]=0 alone doesn't touch device memory.
        d['counts'].copy_to_device(self._zeros_counts)
        # small per-step transfer: inputs only
        d['drive'].copy_to_device(self.drive)
        cursor = self.cursor
        for _ in range(steps):
            s = cursor % DELAY_SLOTS
            f = (cursor + DELAY_TICKS) % DELAY_SLOTS
            _integrate_kernel[self.blocks, THREADS](
                d['v'], d['g'], d['refractory'], d['drive'], d['counts'],
                d['spike_ring'], cursor, f, self.dt)
            _deliver_kernel[self.blocks, THREADS](
                d['cptr'], d['cin'], d['wrev'], d['g'], d['refractory'],
                d['spike_ring'], s)
            _clear_slot_kernel[self.blocks, THREADS](d['spike_ring'], s)
            cursor += 1
        # small per-step transfer: outputs only
        d['counts'].copy_to_host(self.counts)
        self.nactive[0] = int(self.counts.sum())   # display proxy
        self.total_spikes += self.nactive[0]
        self.sim_ms += steps * self.dt
        self.cursor = cursor
        self._last_elapsed = time.perf_counter() - t0
        return cursor

    def device_bytes(self) -> int:
        return sum(a.nbytes for a in self._d.values())


# ---- selftest -----------------------------------------------------------

def _run_cpu_reference(brain, drive, steps):
    """Run the original CPU kernel with the active set seeded from drive."""
    from doom.engine import advance
    if brain.nactive[0] > 0:
        brain.active_flag[brain.active[:brain.nactive[0]]] = 0
        brain.nactive[0] = 0
    driven = np.nonzero(drive)[0].astype(np.int32)
    n = len(driven)
    if n > 0:
        brain.active[:n] = driven
        brain.active_flag[driven] = 1
        brain.nactive[0] = n
    brain.drive[:] = drive
    brain.counts[:] = 0
    brain.cursor = advance(
        brain.ptr, brain.post, brain.weight, brain.v, brain.g,
        brain.refractory, brain.drive, brain.queue, brain.queue_count,
        brain.cursor, steps, brain.dt, brain.counts,
        brain.active, brain.active_flag, brain.nactive)
    return brain.counts.copy()


def run_selftest(doomfly_path, steps=200, seed=42):
    """200 steps on CPU and GPU from identical initial state + drive pattern.

    Prints per-neuron spike-count mismatch stats (max / mean). Zero-ish is
    the pass bar; single-ULP float delivery-order differences can flip a
    threshold for at most a handful of border neurons.
    """
    import sys
    from pathlib import Path
    doomfly_path = Path(doomfly_path).resolve()
    sys.path.insert(0, str(doomfly_path))
    from doom.engine import Brain

    graph_path = doomfly_path / "outputs/doom/malecns_v1/graph.npz"
    print(f"Loading graph {graph_path} ...")
    brain = Brain(graph_path)

    rng = np.random.default_rng(seed)
    n = brain.n
    # identical initial state + a sparse random drive pattern (~0.1% driven)
    brain.v[:] = -52
    drive = np.zeros(n, dtype=np.float32)
    driven = rng.choice(n, size=max(1, n // 1000), replace=False)
    drive[driven] = rng.uniform(5, 30, size=len(driven)).astype(np.float32)

    print("Running CPU reference (numba njit, JIT ~30-60s on first run) ...")
    t0 = time.perf_counter()
    cpu_counts = _run_cpu_reference(brain, drive, steps)
    print(f"  CPU: {time.perf_counter()-t0:.1f}s, "
          f"total spikes={cpu_counts.sum()}")

    print("Building CUDA engine ...")
    gbrain = CudaBrain(brain)
    gbrain.reset_state(v=-52.0)
    gbrain.drive[:] = drive
    print("Running GPU (numba cuda.jit, JIT ~30-60s on first run) ...")
    t0 = time.perf_counter()
    gbrain.advance(steps)
    gpu_counts = gbrain.counts
    print(f"  GPU: {time.perf_counter()-t0:.1f}s, "
          f"total spikes={gpu_counts.sum()}")

    diff = np.abs(cpu_counts.astype(np.int64) - gpu_counts.astype(np.int64))
    mism = int((diff > 0).sum())
    print("\n=== selftest: per-neuron spike-count mismatch over "
          f"{steps} steps ===")
    print(f"  mismatching neurons : {mism} / {n}")
    print(f"  max |diff|          : {int(diff.max())}")
    print(f"  mean |diff|         : {diff.mean():.6g}")
    if mism == 0:
        print("  PASS: exact match")
    elif diff.max() <= 2 and mism <= n // 1000:
        print("  PASS (zero-ish): differences consistent with float32 "
              "delivery-order rounding")
    else:
        print("  FAIL: mismatch too large — investigate")
    return mism, int(diff.max()), float(diff.mean())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CUDA engine selftest")
    p.add_argument("--doomfly", default="../doomfly")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not cuda_available():
        print("ERROR: numba.cuda.is_available() is False — no usable GPU.")
        raise SystemExit(1)
    run_selftest(a.doomfly, a.steps, a.seed)
