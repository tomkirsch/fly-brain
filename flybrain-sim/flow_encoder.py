"""
Synthetic optical flow encoder.

Converts fly velocity + heading into drive currents for T4/T5 and looming
neurons, replacing the webcam + Farneback pipeline.

T4/T5 direction convention (from boat.horse article):
  "front-to-back on the left eye is leftward image motion"
  "T4/T5 subtype a in both eyes gives DNa02 left 26 Hz vs right 2 Hz"

So T4a = front-to-back motion detector.

For a fly at heading θ moving with velocity (vx, vy):
  - forward component: fwd = vx*cos(θ) + vy*sin(θ)
  - lateral component: lat = -vx*sin(θ) + vy*cos(θ)  (positive = rightward)

Left eye sees: front-to-back (T4a) ∝ fwd + lat  (outer flow faster when turning)
Right eye sees: front-to-back (T4a) ∝ fwd - lat

CALIBRATION (boat.horse reference):
  "T4a in both eyes → DNa02 left 26 Hz vs right 2 Hz" is a biological rate.
  In the sim (200 LIF ticks = 20ms per window), 26 Hz = ~0.5 spikes/window.
  With FLOW_GAIN=20 we see ~6 spikes/window (fine for navigation; well below
  the ~9-spike physical max set by the 2.2ms refractory period).
  Do NOT chase the "26" number — tune for stable 3-8 spikes/window instead.
  Use brain_runner.py --calibrate to check.
"""

import numpy as np


# FLOW_GAIN=20 gives dna02 ~6 spikes/200-tick window (≈300 Hz sim-rate, well above
# biological 26 Hz but adequate for navigation). Max achievable ≈ 9 (refractory-limited).
# Lower if nactive blooms past 150k; raise only if dna02 stays at 0.
FLOW_GAIN = 20.0   # mV per unit flow
LOOM_GAIN = 12.0   # mV per unit looming (0-1 range)


class FlowEncoder:
    def __init__(self, groups: dict, n_neurons: int):
        """
        groups: dict from neuron_groups.json, values are lists of internal indices.
        n_neurons: total neuron count (brain.n).
        """
        self.n = n_neurons
        # Convert to numpy arrays for fast indexing
        self._g = {k: np.array(v, dtype=np.int32) for k, v in groups.items()}
        self._report_coverage()

    def _report_coverage(self):
        keys = ["t4a_left", "t4a_right", "lc4_left", "lc4_right", "dna02_left"]
        for k in keys:
            arr = self._g.get(k, np.array([], dtype=np.int32))
            print(f"  FlowEncoder {k}: {len(arr)} neurons")

    def encode(self, vx: float, vy: float, heading: float,
               looming_left: float, looming_right: float) -> np.ndarray:
        """
        Returns drive array shape (n_neurons,) with injection currents in mV.

        vx, vy: velocity in world-pixels/frame
        heading: radians, 0 = right, π/2 = down
        looming_left/right: 0-1, rate of approach to nearest obstacle in each hemifield
        """
        drive = np.zeros(self.n, dtype=np.float32)

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        # Velocity in fly-relative frame
        fwd = vx * cos_h + vy * sin_h      # positive = moving forward
        lat = -vx * sin_h + vy * cos_h     # positive = drifting right

        # Front-to-back flow per eye (positive when fly moves forward)
        left_ftb  = max(0.0,  fwd + lat) * FLOW_GAIN   # T4a/T5a left
        right_ftb = max(0.0,  fwd - lat) * FLOW_GAIN   # T4a/T5a right
        left_btf  = max(0.0, -fwd - lat) * FLOW_GAIN   # T4b/T5b left (reverse flow)
        right_btf = max(0.0, -fwd + lat) * FLOW_GAIN   # T4b/T5b right

        self._inject(drive, "t4a_left",  left_ftb)
        self._inject(drive, "t5a_left",  left_ftb)
        self._inject(drive, "t4a_right", right_ftb)
        self._inject(drive, "t5a_right", right_ftb)
        self._inject(drive, "t4b_left",  left_btf)
        self._inject(drive, "t5b_left",  left_btf)
        self._inject(drive, "t4b_right", right_btf)
        self._inject(drive, "t5b_right", right_btf)

        # Looming detectors
        ll = looming_left  * LOOM_GAIN
        lr = looming_right * LOOM_GAIN
        self._inject(drive, "lc4_left",   ll)
        self._inject(drive, "lplc2_left", ll)
        self._inject(drive, "lc4_right",  lr)
        self._inject(drive, "lplc2_right",lr)

        return drive

    def _inject(self, drive: np.ndarray, key: str, value: float):
        idx = self._g.get(key)
        if idx is not None and len(idx) > 0 and value > 0:
            drive[idx] = value
