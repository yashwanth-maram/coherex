# coherex/integrity/segment_aggregator.py

from collections import deque
from coherex.config import CONFIG


class SegmentIntegrityAggregator:
    """
    Maintains a rolling window of frame-level integrity scores and
    computes segment-level temporal integrity metrics.

    Why segment-level?
    ------------------
    Frame-level scores are inherently noisy — a single blip does not
    constitute a forensic event. Segment aggregation provides:

        * Stability across time
        * Evidence persistence (tampered regions last multiple frames)
        * Volatility signal (erratic scores = unstable scene)

    Window sizing guide:
        150 frames @ 30 FPS  =   5 seconds
        300 frames @ 30 FPS  =  10 seconds
        900 frames @ 30 FPS  =  30 seconds (forensic block)
    """

    def __init__(self, window_size: int = None, config=None):
        """
        Args:
            window_size (int | None):
                Number of frames in the rolling window.
                Defaults to CONFIG.integrity.segment_window_frames.
            config (SystemConfig | None):
                Optional config override. Defaults to global CONFIG.
        """
        cfg = config or CONFIG
        self.window_size = (
            window_size if window_size is not None
            else cfg.integrity.segment_window_frames
        )
        self.scores = deque(maxlen=self.window_size)

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, frame_integrity_score: float):
        """
        Ingest a new frame-level integrity score into the window.
        """
        self.scores.append(frame_integrity_score)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_segment_score(self) -> float:
        """Mean integrity over the current window."""
        if not self.scores:
            return 1.0
        return sum(self.scores) / len(self.scores)

    def get_min_score(self) -> float:
        """Worst (lowest) integrity moment within the current window."""
        if not self.scores:
            return 1.0
        return min(self.scores)

    def get_volatility(self) -> float:
        """Standard deviation of integrity scores in the window."""
        if not self.scores:
            return 0.0
        mean = self.get_segment_score()
        variance = sum((s - mean) ** 2 for s in self.scores) / len(self.scores)
        return variance ** 0.5

    def get_snapshot(self) -> dict:
        """Return all metrics as a single dict for logging or export."""
        return {
            "segment_score":    round(self.get_segment_score(), 4),
            "min_score":        round(self.get_min_score(), 4),
            "volatility":       round(self.get_volatility(), 4),
            "frames_in_window": len(self.scores),
        }
