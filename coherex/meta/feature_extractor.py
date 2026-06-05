# coherex/meta/feature_extractor.py
"""
Feature Consolidation Layer — Baseline Edition
==============================================
Converts raw pipeline outputs into a stable, log-transformed feature
vector suitable for the Random Forest meta-classifier.

Strict Feature Set:
  - motion_min, motion_mean
  - min_integrity
  - volatility
  - anomaly_density
  - perc_below_06
  - max_mcv_log
"""

import math

# ─── Canonical Feature Names (must match training script) ──────────────────
# NOTE: Order must match the vector produced by build_feature_vector
FEATURE_NAMES = [
    "motion_min",
    "motion_mean",
    "min_integrity",
    "volatility",
    "anomaly_density",
    "perc_below_06",
    "max_mcv_log",
]

def build_feature_vector(
    motion_mean: float,
    motion_min: float,
    min_integrity: float,
    volatility: float,
    anomaly_density: float,
    perc_below_06: float,
    max_mcv: float,
) -> list:
    """
    Build the canonical feature vector from raw pipeline outputs.
    Follows the PERFECT EXECUTION PROMPT logic.
    """
    return [
        float(motion_min),
        float(motion_mean),
        float(min_integrity),
        float(volatility),
        float(anomaly_density),
        float(perc_below_06),
        math.log1p(max(0.0, float(max_mcv))),  # max_mcv_log
    ]

def build_feature_dict(
    motion_mean: float,
    motion_min: float,
    min_integrity: float,
    volatility: float,
    anomaly_density: float,
    perc_below_06: float,
    max_mcv: float,
) -> dict:
    vec = build_feature_vector(
        motion_mean=motion_mean,
        motion_min=motion_min,
        min_integrity=min_integrity,
        volatility=volatility,
        anomaly_density=anomaly_density,
        perc_below_06=perc_below_06,
        max_mcv=max_mcv
    )
    return dict(zip(FEATURE_NAMES, vec))
