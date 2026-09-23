"""
2D virtual world for the fly.

Physics: differential drive from left/right DN fire rates.
Looming: rate of approach to nearest wall in left/right visual hemifields.

Coordinate system: x right, y down, heading in radians (0 = right, π/2 = down).
"""

import math
from collections import deque
import numpy as np


SPEED_GAIN   = 25.0   # pixels/sec per normalized spike (dna02 ~6 normalized → ~150px/s)
TURN_GAIN    = 0.75   # rad/sec per spike differential (tuned for ~9x realtime; raise if running faster)
DRAG         = 0.85   # velocity decay per frame (smooths motion)
LOOM_RANGE   = 280.0  # pixels at which looming starts (wider → earlier detection)
WANDER_SPEED = 40.0   # px/sec baseline wander when brain output is silent
WANDER_TURN  = 0.03   # rad/frame random drift
LOOM_TURN    = 2.5    # rad/sec turning bias per unit looming differential in wander mode

# Neural looming escape (LC4/LPLC2 output read from brain)
# Calibration (--calibrate, no looming injected) confirmed LPLC2 baseline = ~2.1
# from T4/T5 background connectivity.  Max near-wall signal = ~3.5.
# Dynamic range is only 1.4 normalized spikes, so we subtract the baseline
# before applying threshold and gain — working with wall-proximity ABOVE noise.
# Tune BRAIN_LOOM_BASELINE if a fresh calibration shows a different floor.
BRAIN_LOOM_BASELINE  = 4.0   # subtract: converts raw count to wall-proximity signal
                             # DNp01 open-field baseline ~4-5 spikes/window (was 2.0 for LPLC2)
BRAIN_LOOM_THRESHOLD = 0.4   # adjusted spikes above baseline; DNp01 quantizes 0→4→8 so
                             # threshold fires when esc = 8 (adj = 4)
BRAIN_LOOM_TURN      = 0.75  # rad/sec per adjusted spike differential
                             # was 3.0 for LPLC2 (adj peak ~1); DNp01 adj peak ~4 → scale ÷4
BRAIN_LOOM_SCATTER   = 8.0   # rad/sec random kick when BOTH eyes above threshold (head-on)
MECH_CONTACT_FRAMES  = 3     # frames of sustained contact for full sensory pressure
_BASELINE_WINDOW     = 200   # open-field samples for adaptive loom baseline

# Neural contact escape via DNg29 — dominant JO-CM downstream target
# (JO-CM → DNg29 weight 157.0, 16 synapses, rank 1 in MaleCNS trace).
# DNg29 baseline is ~0 (silent in free flight); no baseline subtraction needed.
# Injection is one synapse downstream of JO-CM because JO somaSide=nan in MaleCNS.
BRAIN_CONTACT_BASELINE  = 0.0   # DNg29 open-field baseline (expected ~0)
BRAIN_CONTACT_THRESHOLD = 0.4   # adjusted spikes above baseline to trigger escape
BRAIN_CONTACT_TURN      = 1.5   # rad/sec per spike differential (tune after test)


class World:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.margin = 40

        self.turn_gain = TURN_GAIN  # adjustable at runtime (browser slider)

        self.x = float(width / 2)
        self.y = float(height / 2)
        self.heading = np.random.uniform(0, 2 * math.pi)
        self.speed = 30.0   # px/sec kickstart; brain takes over within a few frames
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self._silent_frames = 0
        self._corner_frames = 0
        self._wall_frames = 0
        # Sensory output consumed by the next brain frame.  These are contact
        # channels, not a shortcut to motor control.
        self.mech_left = 0.0
        self.mech_right = 0.0

        # Per-step control decomposition for console/HUD diagnostics. These are
        # world-space translations of neural outputs, not extra control paths.
        self.last_dn_turn      = 0.0
        self.last_loom_turn    = 0.0
        self.last_scatter_turn = 0.0
        self.last_contact_turn = 0.0

        # Sensor outputs (updated each step, read by FlowEncoder)
        self.looming_left  = 0.0
        self.looming_right = 0.0

        # World obstacles: list of circle dicts {cx, cy, r}
        self.obstacles = []

        # Adaptive loom baseline: rolling mean of open-field escape DN samples.
        # Updated only when signal is clearly below threshold (not near a wall).
        # Falls back to BRAIN_LOOM_BASELINE until 50 samples accumulate.
        self._baseline_samples: deque = deque(maxlen=_BASELINE_WINDOW)
        self._loom_baseline: float = BRAIN_LOOM_BASELINE

    def step(self, left_dn_rate: float, right_dn_rate: float, dt: float = 0.020,
             loom_l: float = 0.0, loom_r: float = 0.0,
             contact_l: float = 0.0, contact_r: float = 0.0,
             no_scatter: bool = False):
        """
        Update fly position from descending neuron fire rates.
        left_dn_rate / right_dn_rate: normalized spike counts (~200-tick window)
        dt: seconds per frame (actual wall-clock elapsed, not nominal)
        loom_l / loom_r: normalized LC4/LPLC2 spike counts from brain; drives
                         escape turns when above BRAIN_LOOM_THRESHOLD
        """
        forward_rate = (left_dn_rate + right_dn_rate) / 2.0
        turn_diff    = right_dn_rate - left_dn_rate
        self.last_dn_turn      = 0.0
        self.last_loom_turn    = 0.0
        self.last_scatter_turn = 0.0
        self.last_contact_turn = 0.0

        if forward_rate < 0.01 and abs(turn_diff) < 0.01:
            # Brain is silent — wander with geometric looming-based steering.
            self._silent_frames += 1
            loom_total = (self.looming_left + self.looming_right) / 2
            loom_diff  = self.looming_right - self.looming_left
            turn_bias  = loom_diff * LOOM_TURN * dt
            self.last_loom_turn = turn_bias
            rand_scale = 1.0 + loom_total * 8.0
            self.heading += turn_bias + np.random.uniform(-WANDER_TURN * rand_scale,
                                                           WANDER_TURN * rand_scale)
        else:
            self._silent_frames = 0
            # DN differential drives turning.
            # Suppress turn_dn only when it presses INTO the contact wall.
            # If it would help escape (same sign as away-from-wall direction), let it through.
            # Corner contact (L≈R) leaves turn_dn unsuppressed; corner scatter handles deadlock.
            # The connectome has no DNp01→DNa02 inhibition; this is a sim design choice.
            mech_contact = max(self.mech_left, self.mech_right)
            wall_pressing = (
                (self.mech_right > self.mech_left and turn_diff > 0) or
                (self.mech_left > self.mech_right and turn_diff < 0)
            )
            suppression = mech_contact if wall_pressing else 0.0
            self.last_dn_turn = turn_diff * self.turn_gain * dt * (1.0 - suppression)
            self.heading += self.last_dn_turn
            # Adaptive baseline: sample open-field escape DN rate to track background.
            # Only sample when signal is well below threshold so wall frames don't
            # inflate the baseline.  Mean stabilises after ~200 open-field frames.
            peak_loom = max(loom_l, loom_r)
            if peak_loom - self._loom_baseline < BRAIN_LOOM_THRESHOLD * 0.5:
                self._baseline_samples.append(peak_loom)
                if len(self._baseline_samples) >= 50:
                    self._loom_baseline = float(np.mean(self._baseline_samples))

            # Neural looming escape: subtract adaptive baseline to isolate
            # wall-proximity signal above noise.
            adj_l = max(0.0, loom_l - self._loom_baseline)
            adj_r = max(0.0, loom_r - self._loom_baseline)
            if max(adj_l, adj_r) > BRAIN_LOOM_THRESHOLD:
                # adj_l > adj_r → left wall closer → positive escape → turns right
                escape = (adj_l - adj_r) * BRAIN_LOOM_TURN * dt
                self.last_loom_turn = escape
                # Both eyes above threshold: head-on or symmetric wall — random kick
                if not no_scatter and min(adj_l, adj_r) > BRAIN_LOOM_THRESHOLD:
                    self.last_scatter_turn = np.random.choice([-1.0, 1.0]) * BRAIN_LOOM_SCATTER * dt
                    escape += self.last_scatter_turn
                self.heading += escape
            # Corner-contact scatter: bilateral contact with symmetric DNp01 gives
            # zero adj_l/adj_r and zero turn_loom. One-shot kick on contact onset
            # (frame == MECH_CONTACT_FRAMES) breaks symmetry without repeating.
            # Sim design choice — no neural basis; documented in README.
            if not no_scatter and self.mech_left > 0 and self.mech_right > 0 and self._corner_frames == MECH_CONTACT_FRAMES:
                corner_kick = np.random.choice([-1.0, 1.0]) * BRAIN_LOOM_SCATTER * dt
                self.last_scatter_turn += corner_kick
                self.heading += corner_kick


        # Neural mechanosensory escape via DNg29 — runs regardless of visual drive state.
        # JO-CM → DNg29 is the dominant first-synapse pathway (weight 157, 16 syn).
        # contact_l > contact_r → left wall → positive turn → escapes right.
        adj_cl = max(0.0, contact_l - BRAIN_CONTACT_BASELINE)
        adj_cr = max(0.0, contact_r - BRAIN_CONTACT_BASELINE)
        if max(adj_cl, adj_cr) > BRAIN_CONTACT_THRESHOLD:
            contact_escape = (adj_cl - adj_cr) * BRAIN_CONTACT_TURN * dt
            self.last_contact_turn = contact_escape
            self.heading += contact_escape

        self.heading = self.heading % (2 * math.pi)

        target_speed = forward_rate * SPEED_GAIN if forward_rate >= 0.01 else WANDER_SPEED
        self.speed = self.speed * DRAG + target_speed * (1 - DRAG)

        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Soft wall: clamp position and zero the inward velocity component so the fly
        # doesn't keep trying to push through the margin boundary each frame.
        m = self.margin
        if self.x <= m:
            self.x = m;           self.vx = max(0.0, self.vx)
        elif self.x >= self.width - m:
            self.x = self.width - m;  self.vx = min(0.0, self.vx)
        if self.y <= m:
            self.y = m;           self.vy = max(0.0, self.vy)
        elif self.y >= self.height - m:
            self.y = self.height - m; self.vy = min(0.0, self.vy)

        # Tactile collision responses: mechanosensory feedback when pressing a wall.
        at_h = (self.x <= m) or (self.x >= self.width  - m)
        at_v = (self.y <= m) or (self.y >= self.height - m)
        if at_h and at_v:
            # Corner: both walls simultaneously; expose bilateral contact pressure.
            self._corner_frames += 1
            self._wall_frames = 0
            pressure = min(1.0, self._corner_frames / MECH_CONTACT_FRAMES)
            self.mech_left = pressure
            self.mech_right = pressure
        elif at_h or at_v:
            # Single wall: expose sustained contact to the neural input.
            self._corner_frames = 0
            self._wall_frames += 1
            pressure = min(1.0, self._wall_frames / MECH_CONTACT_FRAMES)
            # Left/right are body-side channels. For top/bottom contact, use
            # the side toward which the fly is facing as a stable 2-D proxy.
            if at_h:
                if self.x <= m:
                    self.mech_left = pressure
                else:
                    self.mech_right = pressure
            else:
                if math.sin(self.heading) <= 0:
                    self.mech_left = pressure
                else:
                    self.mech_right = pressure
        else:
            self._corner_frames = 0
            self._wall_frames = 0
            self.mech_left = 0.0
            self.mech_right = 0.0

        self._update_looming()

    def _update_looming(self):
        # 45° off-center covers the forward hemifield: fly sees walls ahead, not just beside it.
        # Full π FOV per eye so the left eye covers the left 180° and right eye the right 180°.
        left_angle  = self.heading - math.pi / 4
        right_angle = self.heading + math.pi / 4

        dist_l = self._nearest_obstacle_distance(left_angle,  fov=math.pi)
        dist_r = self._nearest_obstacle_distance(right_angle, fov=math.pi)

        self.looming_left  = max(0.0, 1.0 - dist_l / LOOM_RANGE) ** 2
        self.looming_right = max(0.0, 1.0 - dist_r / LOOM_RANGE) ** 2

    def _nearest_obstacle_distance(self, center_angle: float, fov: float) -> float:
        """Minimum distance to any wall or obstacle within fov/2 of center_angle."""
        dists = []

        # Axis-aligned walls as ray cast
        cx, cy = math.cos(center_angle), math.sin(center_angle)
        m = self.margin
        candidates = []
        if abs(cx) > 0.01:
            t = ((self.width - m - self.x) if cx > 0 else (self.x - m)) / abs(cx)
            candidates.append(t)
        if abs(cy) > 0.01:
            t = ((self.height - m - self.y) if cy > 0 else (self.y - m)) / abs(cy)
            candidates.append(t)
        if candidates:
            dists.append(min(candidates))

        # Circle obstacles
        for obs in self.obstacles:
            dx = obs["cx"] - self.x
            dy = obs["cy"] - self.y
            dist = math.sqrt(dx * dx + dy * dy) - obs["r"]
            angle_to = math.atan2(dy, dx)
            angle_diff = abs((angle_to - center_angle + math.pi) % (2*math.pi) - math.pi)
            if angle_diff < fov / 2:
                dists.append(max(1.0, dist))

        return min(dists) if dists else LOOM_RANGE * 2

    def state_dict(self):
        return {
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "heading": round(self.heading, 3),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "speed": round(self.speed, 2),
            "looming_left":  round(self.looming_left, 3),
            "looming_right": round(self.looming_right, 3),
            "dn_turn":      round(self.last_dn_turn,      4),
            "loom_turn":    round(self.last_loom_turn,    4),
            "scatter_turn": round(self.last_scatter_turn, 4),
            "contact_turn": round(self.last_contact_turn, 4),
            "mech_left": round(self.mech_left, 3),
            "mech_right": round(self.mech_right, 3),
            "width": self.width,
            "height": self.height,
            "obstacles": self.obstacles,
            "loom_baseline": round(self._loom_baseline, 3),
            "loom_baseline_n": len(self._baseline_samples),
        }
