# coherex/integrity/fusion_engine.py
"""
Integrity Fusion Engine — Dynamic Reliability-Aware Fusion

Fuses per-track and scene-level coherence signals from multiple
integrity agents into a single, global integrity score.

When a ReliabilityEstimator is provided, weights are dynamically
scaled per frame based on agent reliability:

    EffectiveWeight_i = BaseWeight_i × Reliability_i
    Normalize → Fuse

When no reliability context is supplied, static base weights are
used (backward compatible).
"""

from coherex.config import CONFIG


class IntegrityFusionEngine:
    """
    Score interpretation (from CONFIG.integrity.verdict_thresholds):
        > HIGH      High Integrity       — scene is temporally consistent
        MOD–HIGH    Moderate Risk         — some instability signals present
        < MOD       Integrity Compromised — significant anomaly detected
    """

    def __init__(self, config=None, reliability_estimator=None):
        self._config = config or CONFIG
        w_motion, w_continuity, w_crowd = self._config.integrity.fusion_weights

        self.w_motion = w_motion
        self.w_continuity = w_continuity
        self.w_crowd = w_crowd

        self.reliability = reliability_estimator

        # Last-computed reliabilities (for dashboard / reporting)
        self._last_r_motion = 1.0
        self._last_r_continuity = 1.0
        self._last_r_crowd = 1.0

        self._validate_weights()

    def _validate_weights(self):
        total = self.w_motion + self.w_continuity + self.w_crowd
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Fusion weights must sum to 1.0, got {total:.6f}. "
                f"(w_motion={self.w_motion}, w_continuity={self.w_continuity}, "
                f"w_crowd={self.w_crowd})"
            )

    def fuse(
        self,
        motion_scores: list,
        continuity_scores: list,
        crowd_score: float,
        tracks=None,
        trajectory_store=None,
    ) -> float:
        """
        Compute the global integrity score for a single frame.

        Args:
            motion_scores:      Per-track motion coherence scores.
            continuity_scores:  Per-track lifecycle coherence scores.
            crowd_score:        Scene-level directional coherence.
            tracks:             (optional) Active Track objects for reliability.
            trajectory_store:   (optional) TrajectoryStore for reliability.

        Returns:
            float: Global integrity score in [0.0, 1.0].
        """

        if not motion_scores or not continuity_scores:
            return 1.0

        avg_motion = sum(motion_scores) / len(motion_scores)
        avg_continuity = sum(continuity_scores) / len(continuity_scores)

        # ── Dynamic fusion (reliability-aware) ────────────────────────
        if self.reliability is not None and tracks is not None:
            motion_reliabilities = []
            continuity_reliabilities = []

            for track in tracks:
                state = (
                    trajectory_store.get_state(track.track_id)
                    if trajectory_store is not None else None
                )
                r_m = self.reliability.motion_reliability(track, state)
                r_c = self.reliability.continuity_reliability(track)

                motion_reliabilities.append(r_m)
                continuity_reliabilities.append(r_c)

            # Mean reliability across all tracks
            r_motion = (
                sum(motion_reliabilities) / len(motion_reliabilities)
                if motion_reliabilities else 0.0
            )
            r_continuity = (
                sum(continuity_reliabilities) / len(continuity_reliabilities)
                if continuity_reliabilities else 0.0
            )
            r_crowd = self.reliability.crowd_reliability(len(tracks))

            # Store for reporting
            self._last_r_motion = r_motion
            self._last_r_continuity = r_continuity
            self._last_r_crowd = r_crowd

            # Scale base weights by reliability
            ew_motion = self.w_motion * r_motion
            ew_continuity = self.w_continuity * r_continuity
            ew_crowd = self.w_crowd * r_crowd

            weight_sum = ew_motion + ew_continuity + ew_crowd
            if weight_sum == 0:
                return 1.0  # no reliable signal → assume ok

            ew_motion /= weight_sum
            ew_continuity /= weight_sum
            ew_crowd /= weight_sum

            integrity = (
                ew_motion * avg_motion
                + ew_continuity * avg_continuity
                + ew_crowd * crowd_score
            )

        else:
            # ── Static fusion (backward compatible) ───────────────────
            self._last_r_motion = 1.0
            self._last_r_continuity = 1.0
            self._last_r_crowd = 1.0

            integrity = (
                self.w_motion * avg_motion
                + self.w_continuity * avg_continuity
                + self.w_crowd * crowd_score
            )

        return max(0.0, min(1.0, integrity))

    def get_last_reliabilities(self):
        """
        Returns the most recently computed reliability scores.
        Useful for dashboard display and report generation.

        Returns:
            tuple: (R_motion, R_continuity, R_crowd)
        """
        return (
            self._last_r_motion,
            self._last_r_continuity,
            self._last_r_crowd,
        )

    def interpret(self, integrity_score: float) -> str:
        """
        Map a numeric integrity score to a human-readable label.

        Returns:
            str: 'HIGH' | 'MODERATE' | 'COMPROMISED'
        """
        high, moderate = self._config.integrity.verdict_thresholds
        if integrity_score > high:
            return "HIGH"
        elif integrity_score >= moderate:
            return "MODERATE"
        else:
            return "COMPROMISED"
