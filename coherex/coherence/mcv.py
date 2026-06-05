import numpy as np


class MotionCoherenceEngine:
    """
    Computes continuous Motion Coherence Value (MCV)
    from trajectory motion signals.

    Pure engine — does NOT import CONFIG.
    Parameters are injected by the calling agent.
    """

    def __init__(self, window_size=3, weights=(0.4, 0.3, 0.3), eps=1e-6, noise_floor=1.0):
        self.window_size = window_size
        self.w_speed, self.w_acc, self.w_angle = weights
        self.eps = eps
        self.noise_floor = noise_floor

    def _z_score(self, values):
        """
        Compute robust z-score for the latest value.
        Uses a noise floor to prevent score explosion on tiny jitter.
        """
        if len(values) < self.window_size:
            return 0.0

        vals = list(values)
        window = np.array(vals[-self.window_size:])
        
        # Compare latest value against the mean/std of PREVIOUS values
        # to avoid the shill-effect of the outlier pulling the distribution.
        previous = window[:-1]
        mean = np.mean(previous)
        std = np.std(previous)
        
        # Apply noise floor: if natural variations are tiny, we don't
        # want the z-score to explode on a 0.5px jitter.
        effective_std = max(std, self.noise_floor) + self.eps

        return (window[-1] - mean) / effective_std

    def compute(self, trajectory_state, track_confidence=1.0):
        """
        Compute Motion Coherence Value for one track.
        """

        z_speed = self._z_score(trajectory_state.speeds)
        z_acc = self._z_score(trajectory_state.accelerations)
        z_angle = self._z_score(trajectory_state.angles)

        mcv = (
            self.w_speed * z_speed
            + self.w_acc * z_acc
            + self.w_angle * z_angle
        )

        return mcv
