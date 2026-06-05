# coherex/integrity/continuity_agent.py

from coherex.config import CONFIG
from coherex.integrity.base_agent import IntegrityAgent


class ContinuityAgent(IntegrityAgent):
    """
    Evaluates tracking lifecycle stability and identity continuity.

    Penalises:
        1. Track termination         → immediate 0.0 (track no longer exists)
        2. Active tamper latch        → immediate 0.0 (identity break in progress)
        3. Accumulated misses         → graduated penalty (more misses = lower score)
        4. Dormant state              → fixed penalty (track lost contact)
        5. Repeated re-attachments   → graduated penalty (identity instability)

    Returns:
        coherence_score (float) in [0.0, 1.0]
            1.0 = perfectly stable, continuously matched track
            0.0 = terminated or actively tampered
    """

    def __init__(
        self,
        miss_penalty_scale: float = None,
        dormant_penalty: float = None,
        reattach_penalty: float = None,
        max_reattach_penalty: float = None,
        config=None,
    ):
        cfg = config or CONFIG
        cc = cfg.integrity.continuity
        self.miss_penalty_scale = (
            miss_penalty_scale if miss_penalty_scale is not None
            else cc.miss_penalty_scale
        )
        self.dormant_penalty = (
            dormant_penalty if dormant_penalty is not None
            else cc.dormant_penalty
        )
        self.reattach_penalty = (
            reattach_penalty if reattach_penalty is not None
            else cc.reattach_penalty
        )
        self.max_reattach_penalty = (
            max_reattach_penalty if max_reattach_penalty is not None
            else cc.max_reattach_penalty
        )

    def evaluate(self, track, frame_id: int) -> float:
        """
        Compute the continuity coherence score for a single track.

        Args:
            track (Track):   The track object to evaluate.
            frame_id (int):  The current frame index.

        Returns:
            float: Coherence score in [0.0, 1.0].
        """

        # ── Hard overrides (immediate disqualification) ──────────────────────

        # 1. Terminated track = zero integrity — it no longer participates
        if track.state.name == "TERMINATED":
            return 0.0

        # 2. Tamper latch is active — identity break locked in for this window
        if frame_id <= getattr(track, "tamper_until_frame", -1):
            return 0.0

        # ── Graduated penalties ──────────────────────────────────────────────

        score = 1.0

        # 3. Miss penalty: proportional to accumulated missed detections
        miss_penalty = min(track.misses / self.miss_penalty_scale, 1.0)
        score -= miss_penalty

        # 4. Dormant penalty: track is active but has temporarily lost contact
        if track.state.name == "DORMANT":
            score -= self.dormant_penalty

        # 5. Reattachment penalty: each successful dormant re-link is suspicious
        reattach_count = getattr(track, "reattach_count", 0)
        if reattach_count > 0:
            ra_penalty = min(
                reattach_count * self.reattach_penalty,
                self.max_reattach_penalty
            )
            score -= ra_penalty

        return max(0.0, score)
