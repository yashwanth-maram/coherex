# coherex/integrity/crowd_agent.py

import numpy as np
from coherex.config import CONFIG
from coherex.integrity.base_agent import IntegrityAgent


class CrowdAgent(IntegrityAgent):
    """
    Evaluates relational motion consistency across all active tracks.

    Instead of analysing tracks in isolation, this agent asks:

        "Are people moving coherently relative to each other?"

    In normal scenes, crowd motion has directional structure — even in
    complex scenes, nearby tracks share broad directional tendencies.
    Tampering (e.g. frame injection, speed manipulation, ghost tracks)
    tends to break this relative symmetry.

    Algorithm
    ---------
    1. Collect the normalised velocity unit-vectors of all tracks that
       have meaningful motion (speed > min_speed_threshold).
    2. Compute the scene's mean motion direction.
    3. Compute each track's angular deviation from that mean.
    4. Average the deviations → scene-level directional variance.
    5. Map variance → coherence score in [0, 1].

    Returns
    -------
    coherence_score (float) in [0.0, 1.0]
        1.0 = all tracks moving in a coherent shared direction
        0.0 = maximum directional chaos (all tracks diverging)
    """

    def __init__(
        self,
        min_speed_threshold: float = None,
        min_tracks: int = None,
        max_deviation: float = None,
        config=None,
    ):
        cfg = config or CONFIG
        crc = cfg.integrity.crowd
        self.min_speed_threshold = (
            min_speed_threshold if min_speed_threshold is not None
            else crc.min_speed_threshold
        )
        self.min_tracks = (
            min_tracks if min_tracks is not None
            else crc.min_tracks
        )
        self.max_deviation = (
            max_deviation if max_deviation is not None
            else crc.max_deviation
        )

    def evaluate(self, tracks) -> float:
        """
        Compute crowd-level motion coherence from a list of tracks.

        Args:
            tracks (list[Track]):
                All currently active (NEW / ACTIVE) tracks in the frame.
                Each track must expose .vx and .vy attributes.

        Returns:
            float: Coherence score in [0.0, 1.0].
        """

        unit_vectors = self._collect_unit_vectors(tracks)

        if len(unit_vectors) < self.min_tracks:
            return 1.0

        vstack = np.array(unit_vectors)           # shape (N, 2)
        mean_dir = np.mean(vstack, axis=0)        # scene-level mean direction

        # Mean deviation of each unit vector from the scene mean direction
        deviations = np.linalg.norm(vstack - mean_dir, axis=1)
        avg_deviation = float(np.mean(deviations))

        # Normalise: 0.0 deviation → score 1.0;  max_deviation → score 0.0
        normalized = min(avg_deviation / self.max_deviation, 1.0)
        coherence = 1.0 - normalized

        return max(0.0, coherence)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _collect_unit_vectors(self, tracks) -> list:
        """
        Extract normalised velocity unit-vectors from moving tracks.
        """
        unit_vectors = []

        for track in tracks:
            vx = getattr(track, "vx", 0.0)
            vy = getattr(track, "vy", 0.0)

            speed = np.sqrt(vx * vx + vy * vy)
            if speed >= self.min_speed_threshold:
                unit_vectors.append([vx / speed, vy / speed])

        return unit_vectors
