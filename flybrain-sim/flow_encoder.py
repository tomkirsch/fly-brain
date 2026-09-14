"""
Scene-based optical flow encoder.

Replaces the analytical fwd+lat formula which gave lat≈0 always
(velocity always aligned with heading → no genuine L/R asymmetry).

This version ray-casts the 2D arena from the fly's current position:

  - NUM_RAYS evenly spaced across 360°, 5° per column.
  - Each ray at world angle φ measures distance d(φ) to nearest wall/obstacle.
  - Angular velocity (flow): ω(φ) = speed·sin(heading−φ) / d(φ)
    > 0 on left eye (counterclockwise = front-to-back), < 0 on right

Hemifields (relative to heading):
  Left eye  : offset ∈ (180°, 360°)  i.e., the left semicircle
  Right eye : offset ∈ [0°, 180°]    i.e., the right semicircle (incl. forward)

T4a/T5a   → front-to-back flow per eye (sum of positive ω on left, negative on right)
T4b/T5b   → back-to-front flow (opposite direction)
LC4/LPLC2 → ray-cast expansion signal: v_radial = speed·cos(heading−φ), max 0 → toward

Asymmetry example: fly near the top wall while heading right.
  "Up" rays (offset ~270°) are in the left eye and hit the top wall at ~40 px.
  "Down" rays (offset ~90°) are in the right eye and hit the bottom wall at ~480 px.
  Left T4a/T5a signal is 9× right → genuine DNa02 differential.

LOOMING: cos(heading−φ) is the complement of T4a's sin(heading−φ).  T4a is blind
  to head-on (v_perp=0); expansion is maximum head-on.  Same ray-cast infrastructure,
  one formula swap — no geometric scripting.

Gains:
  The old FLOW_GAIN=20 was calibrated for the analytical formula (left_ftb = speed).
  The polar-scan raw sum at baseline is ~8× smaller, so FLOW_GAIN=150 is the new
  starting point. Re-tune with brain_runner.py --calibrate (target: 3–8 spikes/window).
"""

import math
import numpy as np


# ── Tuning ────────────────────────────────────────────────────────────────────
# Re-tune with --calibrate after any world-size or ray-count change.
# Target: dna02 at 3–8 spikes/200-tick window; nactive < 100k.
FLOW_GAIN    = 150.0  # mV per unit flow ray-sum  (up from 20; new formula is ~8× smaller)
LOOM_GAIN    = 1.0    # mV per unit expansion ray-sum (start point; retune with --calibrate)
MECH_GAIN    = 20.0   # mV per normalized contact pressure; calibrated sparse-response test
CONTACT_GAIN = 300.0  # mV per normalized contact pressure; direct DNg29 injection
                      # bypasses bilateral JO pool (JO somaSide=nan in MaleCNS)
                      # 300 needed for reliable DNg29 firing (~50% at 200, higher at 300)

NUM_RAYS = 72       # panoramic columns; 360/72 = 5° per ray


class FlowEncoder:
    def __init__(self, groups: dict, n_neurons: int):
        self.n = n_neurons
        self._g = {k: np.array(v, dtype=np.int32) for k, v in groups.items()}
        # Precomputed offsets [0, 2π) so angles = heading + offsets each frame
        self._offsets = np.linspace(0, 2 * math.pi, NUM_RAYS, endpoint=False)
        # Precomputed hemifield masks (independent of heading because offsets are relative)
        # offset=0 is forward → right eye; offset>180° is left eye
        rel = (self._offsets + math.pi) % (2 * math.pi) - math.pi
        self._left_mask  = (rel < 0)
        self._right_mask = ~self._left_mask
        self._report_coverage()

    def _report_coverage(self):
        keys = ["t4a_left", "t4a_right", "lc4_left", "lc4_right", "dna02_left"]
        for k in keys:
            arr = self._g.get(k, np.array([], dtype=np.int32))
            print(f"  FlowEncoder {k}: {len(arr)} neurons")

    def encode(self, x: float, y: float, vx: float, vy: float, heading: float,
               world_width: float = 800, world_height: float = 600,
               margin: float = 40.0, obstacles: list = None,
               mech_l: float = 0.0, mech_r: float = 0.0) -> np.ndarray:
        """
        Returns drive array shape (n_neurons,) with injection currents in mV.

        x, y              : fly world position
        vx, vy            : velocity in world-pixels/sec
        heading           : radians, 0 = right, π/2 = down
        world_width/height, margin : arena geometry (must match World init)
        obstacles         : list of {cx, cy, r} dicts (from World.obstacles)
        mech_l, mech_r     : contact pressure from left/right body side, 0..1.
                             This is injected into annotated mechanosensory
                             neurons, not directly into motor neurons.
        """
        speed = math.sqrt(vx * vx + vy * vy)

        angles = heading + self._offsets          # absolute world angles for each ray
        d = self._ray_distances(x, y, angles,
                                world_width, world_height, margin,
                                obstacles or [])

        # Per-column translational optic flow
        # v_perp > 0  → counterclockwise → front-to-back for left eye
        # v_perp < 0  → clockwise        → front-to-back for right eye
        v_perp = speed * np.sin(heading - angles)
        inv_d  = 1.0 / d

        lm = self._left_mask
        rm = self._right_mask

        # ── T4a/T5a (front-to-back) ────────────────────────────────────────────
        left_ftb  = float(np.dot(lm, np.maximum(0.0,  v_perp)  * inv_d))
        right_ftb = float(np.dot(rm, np.maximum(0.0, -v_perp)  * inv_d))

        # ── T4b/T5b (back-to-front) ────────────────────────────────────────────
        left_btf  = float(np.dot(lm, np.maximum(0.0, -v_perp)  * inv_d))
        right_btf = float(np.dot(rm, np.maximum(0.0,  v_perp)  * inv_d))

        # ── LC4/LPLC2 (expansion / looming) ───────────────────────────────────
        # v_radial > 0 → moving toward that column → expansion signal
        v_radial  = speed * np.cos(heading - angles)
        expansion = np.maximum(0.0, v_radial) * inv_d
        left_loom  = float(np.dot(self._left_mask,  expansion))
        right_loom = float(np.dot(self._right_mask, expansion))

        drive = np.zeros(self.n, dtype=np.float32)
        self._inject(drive, "t4a_left",    left_ftb   * FLOW_GAIN)
        self._inject(drive, "t5a_left",    left_ftb   * FLOW_GAIN)
        self._inject(drive, "t4a_right",   right_ftb  * FLOW_GAIN)
        self._inject(drive, "t5a_right",   right_ftb  * FLOW_GAIN)
        self._inject(drive, "t4b_left",    left_btf   * FLOW_GAIN)
        self._inject(drive, "t5b_left",    left_btf   * FLOW_GAIN)
        self._inject(drive, "t4b_right",   right_btf  * FLOW_GAIN)
        self._inject(drive, "t5b_right",   right_btf  * FLOW_GAIN)
        self._inject(drive, "lc4_left",    left_loom  * LOOM_GAIN)
        self._inject(drive, "lplc2_left",  left_loom  * LOOM_GAIN)
        self._inject(drive, "lc4_right",   right_loom * LOOM_GAIN)
        self._inject(drive, "lplc2_right", right_loom * LOOM_GAIN)
        self._inject(drive, "mech_left",    max(0.0, mech_l) * MECH_GAIN)
        self._inject(drive, "mech_right",   max(0.0, mech_r) * MECH_GAIN)
        self._inject(drive, "dng29_left",   max(0.0, mech_l) * CONTACT_GAIN)
        self._inject(drive, "dng29_right",  max(0.0, mech_r) * CONTACT_GAIN)

        return drive

    def _ray_distances(self, x: float, y: float, angles: np.ndarray,
                       W: float, H: float, m: float,
                       obstacles: list) -> np.ndarray:
        """
        Cast rays from (x, y) in each direction.
        Returns distances to the nearest wall or obstacle, clipped to ≥ 1.0.
        """
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        d = np.full(len(angles), 1e6, dtype=np.float64)

        # Axis-aligned arena walls (boundary at margin m)
        _wall(d, cos_a,  W - m - x)   # right wall:  x = W-m
        _wall(d, cos_a,  m - x)       # left wall:   x = m
        _wall(d, sin_a,  H - m - y)   # bottom wall: y = H-m
        _wall(d, sin_a,  m - y)       # top wall:    y = m

        # Circular obstacles
        for obs in obstacles:
            dx = obs["cx"] - x
            dy = obs["cy"] - y
            b = -(cos_a * dx + sin_a * dy) * 2
            c = dx * dx + dy * dy - obs["r"] * obs["r"]
            disc = b * b - 4.0 * c
            hit = disc >= 0
            sq = np.where(hit, np.sqrt(np.maximum(0.0, disc)), 0.0)
            t1 = np.where(hit, (-b - sq) * 0.5, 1e6)
            t2 = np.where(hit, (-b + sq) * 0.5, 1e6)
            t_near = np.where(t1 > 1e-3, t1, np.where(t2 > 1e-3, t2, 1e6))
            np.minimum(d, np.where(hit & (t_near > 0), t_near, 1e6), out=d)

        return np.maximum(d, 1.0)

    def _inject(self, drive: np.ndarray, key: str, value: float):
        idx = self._g.get(key)
        if idx is not None and len(idx) > 0 and value > 0:
            drive[idx] = value


def _wall(d: np.ndarray, component: np.ndarray, signed_dist: float):
    """
    Update minimum distance for an axis-aligned wall.

    t = signed_dist / component.  Valid (positive) only when both have the
    same sign — i.e., ray is pointing toward that wall from inside the arena.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(component) > 1e-6, signed_dist / component, 1e6)
    np.minimum(d, np.where(t > 1e-3, t, 1e6), out=d)
