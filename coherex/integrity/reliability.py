# coherex/integrity/reliability.py
"""
Agent Reliability Estimation

Each integrity agent self-estimates its reliability for the current
frame context.  The fusion engine uses these reliability scores to
dynamically scale agent weights:

    EffectiveWeight_i = BaseWeight_i × Reliability_i
    Normalize → Fuse

This eliminates the static-weight limitation and makes CoheRex
adaptive to scene complexity, track maturity, and crowd density.

Reference:
    "Reliability-Aware Multi-Agent Temporal Integrity Fusion"
"""


class ReliabilityEstimator:
    """
    Computes per-agent reliability scores ∈ [0, 1] using clean,
    defensible heuristics.

    Methods return 0.0 when an agent is completely unreliable
    and 1.0 when it is fully trustworthy.
    """

    # ── Motion Agent Reliability ──────────────────────────────────────

    def motion_reliability(self, track, trajectory_state):
        """
        MotionAgent is unreliable when:
          - No trajectory history exists
          - Track has insufficient speed/acceleration samples
          - Track confidence is low (new / uncertain detection)

        Formula:
            R = min(1.0, len(speeds) / required_window) × confidence
        """
        if trajectory_state is None:
            return 0.0

        history_factor = min(
            1.0,
            len(trajectory_state.speeds) / 5
        )

        return history_factor * track.confidence

    # ── Continuity Agent Reliability ──────────────────────────────────

    def continuity_reliability(self, track):
        """
        ContinuityAgent is unreliable for very young tracks that
        haven't had time to accumulate lifecycle evidence.

        Formula:
            R = min(1.0, track.age / 10)
        """
        return min(1.0, track.age / 10)

    # ── Crowd Agent Reliability ───────────────────────────────────────

    def crowd_reliability(self, valid_tracks_count):
        """
        CrowdAgent is unreliable when the scene has too few moving
        tracks to establish directional consensus.

        Formula:
            R = min(1.0, valid_tracks_count / 5)
        """
        return min(1.0, valid_tracks_count / 5)
