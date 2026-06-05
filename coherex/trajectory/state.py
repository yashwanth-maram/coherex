# coherex/trajectory/state.py

import numpy as np
from collections import deque


class TrajectoryState:
    """
    Stores per-track temporal motion state.
    """

    def __init__(self, track_id, max_history=10):
        self.track_id = track_id

        self.positions = deque(maxlen=max_history)
        self.velocities = deque(maxlen=max_history)
        self.speeds = deque(maxlen=max_history)
        self.accelerations = deque(maxlen=max_history)
        self.angles = deque(maxlen=max_history)
        self.total_age = 0

    def update(self, x, y, vx, vy):
        """
        Update trajectory state with new motion values.
        """
        self.total_age += 1

        # Position
        self.positions.append(np.array([x, y]))

        # Velocity
        vel = np.array([vx, vy])
        self.velocities.append(vel)

        # Speed
        speed = np.linalg.norm(vel)
        self.speeds.append(speed)

        # Acceleration
        if len(self.velocities) > 1:
            acc = self.velocities[-1] - self.velocities[-2]
            self.accelerations.append(np.linalg.norm(acc))
        else:
            self.accelerations.append(0.0)

        # Direction change (angle)
        if len(self.velocities) > 1:
            v1 = self.velocities[-2]
            v2 = self.velocities[-1]

            norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_prod > 0:
                cos_theta = np.clip(np.dot(v1, v2) / norm_prod, -1.0, 1.0)
                angle = np.arccos(cos_theta)
            else:
                angle = 0.0

            self.angles.append(angle)
        else:
            self.angles.append(0.0)
