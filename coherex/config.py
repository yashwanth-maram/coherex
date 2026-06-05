# coherex/config.py
"""
CoheRex — Deterministic Configuration System

Single source of truth for ALL thresholds, weights, and parameters.
frozen=True ensures immutability — config cannot change mid-execution.

Usage:
    from coherex.config import CONFIG

    CONFIG.detection.confidence_threshold   # 0.4
    CONFIG.integrity.verdict_thresholds     # (0.85, 0.60)
    CONFIG.to_dict()                        # serialize for report inclusion
"""

from dataclasses import dataclass, asdict
from typing import Tuple


# ─── Detection ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionConfig:
    """Object detection parameters."""
    confidence_threshold: float = 0.4
    person_class_id: int = 0


# ─── Tracking ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackingConfig:
    """Multi-object tracking and identity parameters."""
    max_association_distance: float = 50.0
    max_misses: int = 5
    max_dormant_frames: int = 30
    tamper_latch_frames: int = 30
    appearance_similarity_threshold: float = 0.6
    identity_area_range: Tuple[float, float] = (0.6, 1.6)
    identity_ratio_threshold: float = 0.5
    max_trajectory_history: int = 10


# ─── Motion Coherence ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MotionConfig:
    """Motion coherence analysis parameters."""
    mcv_window_size: int = 8
    mcv_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)   # speed, accel, angle
    mcv_noise_floor: float = 1.5
    max_expected_mcv: float = 1.8
    interpreter_low: float = 0.5    # MCV < low  → COHERENT
    interpreter_high: float = 1.8   # MCV ≥ high → INCONSISTENT


# ─── Integrity Agents ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContinuityConfig:
    """Track lifecycle stability parameters."""
    miss_penalty_scale: float = 10.0
    dormant_penalty: float = 0.2
    reattach_penalty: float = 0.15
    max_reattach_penalty: float = 0.45


@dataclass(frozen=True)
class CrowdConfig:
    """Crowd-level directional coherence parameters."""
    min_speed_threshold: float = 0.5
    min_tracks: int = 2
    max_deviation: float = 1.2


@dataclass(frozen=True)
class IntegrityConfig:
    """Multi-agent integrity fusion parameters."""
    fusion_weights: Tuple[float, float, float] = (0.7, 0.3, 0.0)  # motion, continuity, crowd
    segment_window_frames: int = 150
    verdict_thresholds: Tuple[float, float] = (0.85, 0.60)
    # HIGH       > 0.85
    # MODERATE     0.60–0.85
    # COMPROMISED < 0.60
    continuity: ContinuityConfig = ContinuityConfig()
    crowd: CrowdConfig = CrowdConfig()


# ─── Root Config ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemConfig:
    """Root configuration — single source of truth."""
    version: str = "CoheRex-Integrity v1.0"
    detection: DetectionConfig = DetectionConfig()
    tracking: TrackingConfig = TrackingConfig()
    motion: MotionConfig = MotionConfig()
    integrity: IntegrityConfig = IntegrityConfig()

    def to_dict(self) -> dict:
        """Serialize full config for inclusion in forensic reports."""
        return asdict(self)


# ─── Global Default ──────────────────────────────────────────────────────

CONFIG = SystemConfig()
