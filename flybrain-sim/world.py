"""
2D virtual world for the fly.

Physics: differential drive from left/right DN fire rates.
Looming: rate of approach to nearest wall in left/right visual hemifields.

Coordinate system: x right, y down, heading in radians (0 = right, π/2 = down).
"""

import math
import numpy as np


SPEED_GAIN  = 300.0  # pixels/sec per Hz of forward DN
TURN_GAIN   = 0.5    # rad/sec per Hz differential
DRAG        = 0.85   # velocity decay per frame (smooths motion)
LOOM_RANGE  = 180.0  # pixels at which looming starts
WANDER_SPEED = 40.0  # px/sec baseline wander when brain output is silent
WANDER_TURN  = 0.03  # rad/frame random drift


class World:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.margin = 40

        self.x = float(width / 2)
        self.y = float(height / 2)
        self.heading = np.random.uniform(0, 2 * math.pi)
        self.speed = 30.0   # px/sec kickstart; brain takes over within a few frames
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed
        self._silent_frames = 0

        # Sensor outputs (updated each step, read by FlowEncoder)
        self.looming_left  = 0.0
        self.looming_right = 0.0

        # World obstacles: list of circle dicts {cx, cy, r}
        self.obstacles = []

    def step(self, left_dn_rate: float, right_dn_rate: float, dt: float = 0.020):
        """
        Update fly position from descending neuron fire rates.
        left_dn_rate / right_dn_rate: Hz (spike counts / frame, not normalized yet)
        dt: seconds per frame (default 20ms)
        """
        forward_rate = (left_dn_rate + right_dn_rate) / 2.0
        turn_diff    = right_dn_rate - left_dn_rate

        if forward_rate < 0.01 and abs(turn_diff) < 0.01:
            # Brain is silent — wander so optical flow keeps feeding the network
            self._silent_frames += 1
            self.heading += np.random.uniform(-WANDER_TURN, WANDER_TURN)
            target_speed = WANDER_SPEED
        else:
            self._silent_frames = 0
            self.heading += turn_diff * TURN_GAIN * dt

        self.heading  = self.heading % (2 * math.pi)

        target_speed = forward_rate * SPEED_GAIN if forward_rate >= 0.01 else WANDER_SPEED
        self.speed = self.speed * DRAG + target_speed * (1 - DRAG)

        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Hard wall bounce
        m = self.margin
        if self.x < m:
            self.x = m
            self.vx =  abs(self.vx)
            self.heading = math.atan2(self.vy, abs(self.vx))
        elif self.x > self.width - m:
            self.x = self.width - m
            self.vx = -abs(self.vx)
            self.heading = math.atan2(self.vy, -abs(self.vx))
        if self.y < m:
            self.y = m
            self.vy =  abs(self.vy)
            self.heading = math.atan2(abs(self.vy), self.vx)
        elif self.y > self.height - m:
            self.y = self.height - m
            self.vy = -abs(self.vy)
            self.heading = math.atan2(-abs(self.vy), self.vx)

        self._update_looming()

    def _update_looming(self):
        left_angle  = self.heading - math.pi / 2
        right_angle = self.heading + math.pi / 2

        dist_l = self._nearest_obstacle_distance(left_angle,  fov=math.pi * 0.6)
        dist_r = self._nearest_obstacle_distance(right_angle, fov=math.pi * 0.6)

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
            "width": self.width,
            "height": self.height,
            "obstacles": self.obstacles,
        }
