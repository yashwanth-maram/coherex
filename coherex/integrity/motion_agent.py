# coherex/integrity/motion_agent.py

from coherex.config import CONFIG
from coherex.integrity.base_agent import IntegrityAgent
from coherex.coherence.mcv import MotionCoherenceEngine


class MotionAgent(IntegrityAgent):
    """
    Wraps the MotionCoherenceEngine and injects CONFIG parameters.

    This agent is the bridge between CONFIG and the pure MCVEngine.
    The engine itself has no CONFIG dependency — all values are
    passed explicitly via constructor injection.

    Scoring:
        High score (→ 1.0) = stable, expected motion
        Low score  (→ 0.0) = anomalous, inconsistent motion
    """

    def __init__(self, config=None):
        cfg = config or CONFIG
        self.mcv_engine = MotionCoherenceEngine(
            window_size=cfg.motion.mcv_window_size,
            weights=cfg.motion.mcv_weights,
            noise_floor=getattr(cfg.motion, "mcv_noise_floor", 1.0)
        )
        self.max_expected = cfg.motion.max_expected_mcv

    def evaluate(self, trajectory_state, track_confidence, track=None, current_frame=0):
        """
        Compute the motion coherence score for a single track.

        Args:
            trajectory_state (TrajectoryState | None):
                The trajectory state object for the track.
                If None (new track with no history),
                the track is assumed coherent (score = 1.0).
            track_confidence (float):
                Detection confidence ∈ [0, 1].

        Returns:
            float: Coherence score ∈ [0.0, 1.0].
        """

        if trajectory_state is None:
            return 1.0

        mcv = self.mcv_engine.compute(trajectory_state, track_confidence)
        
        # Take absolute MCV because any deviance (faster or slower) is an anomaly
        normalized = min(abs(mcv) / self.max_expected, 1.0)
        coherence = 1.0 - normalized

        # --- Motion-Triggered Latch ---
        # If we detect a sharp physics violation, force a blackout for several frames.
        if track:
            # If current physics is violent (< 0.2), trigger/extend latch
            if coherence < 0.2:
                # 10 frame penalty
                track.latch_tampering(current_frame, 10)
            
            # If track is currently latched, force integrity to 0
            if track.tamper_until_frame >= current_frame:
                coherence = 0.0

        return max(0.0, coherence)
