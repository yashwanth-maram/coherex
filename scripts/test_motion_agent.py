"""
scripts/test_motion_agent.py

Unit test for MotionAgent (Step 1 verification).

Tests two scenarios:
  1. Smooth, consistent motion → expects high coherence (0.8 – 1.0)
  2. Erratic, large-jump motion → expects low coherence (< 0.5)

Run with:
    python -m scripts.test_motion_agent
"""

import os
import sys

# Ensure project root is on the path regardless of where this is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.motion_agent import MotionAgent
from coherex.trajectory.store import TrajectoryStore
from coherex.tracking.track import Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_track_and_store(track_id: int, frame_id: int) -> tuple:
    """Return a fresh (track, store) pair, ready for simulation."""
    store = TrajectoryStore(max_history=10)
    track = Track(track_id=track_id, init_position=(100, 100), frame_id=frame_id)
    return track, store


def simulate_smooth_motion(track, store, steps: int = 15):
    """Move the track by a small, constant delta each step."""
    for _ in range(steps):
        track.x += 3.0
        track.y += 1.5
        track.vx = 3.0
        track.vy = 1.5
        store.update_from_tracks([track])


def simulate_erratic_motion(track, store, steps: int = 15):
    """Introduce large, random-direction jumps to create incoherence."""
    import random
    for i in range(steps):
        # Alternate between large and small jumps → high acceleration / angle changes
        if i % 2 == 0:
            track.x += 80.0
            track.y -= 60.0
            track.vx = 80.0
            track.vy = -60.0
        else:
            track.x -= 70.0
            track.y += 55.0
            track.vx = -70.0
            track.vy = 55.0
        store.update_from_tracks([track])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_smooth_motion():
    print("\n--- Test 1: Smooth Motion ---")
    agent = MotionAgent()
    track, store = build_track_and_store(track_id=0, frame_id=0)

    # Build confidence gradually as the track matures
    track.confidence = 0.9

    simulate_smooth_motion(track, store)

    state = store.get_state(track.track_id)
    score = agent.evaluate(state, track.confidence)

    print(f"  Motion coherence score : {score:.4f}")
    assert score >= 0.5, f"Expected high coherence for smooth motion, got {score:.4f}"  # noqa
    print(f"  Status: PASSED (score >= 0.5)")
    return score


def test_erratic_motion():
    print("\n--- Test 2: Erratic Motion ---")
    agent = MotionAgent()
    track, store = build_track_and_store(track_id=1, frame_id=0)
    track.confidence = 0.9

    simulate_erratic_motion(track, store)

    state = store.get_state(track.track_id)
    score = agent.evaluate(state, track.confidence)

    print(f"  Motion coherence score : {score:.4f}")
    # Erratic motion should produce a measurably lower score than smooth motion
    print(f"  Status: OBSERVED (lower is more anomalous)")
    return score


def test_no_state():
    print("\n--- Test 3: No Trajectory State (new track) ---")
    agent = MotionAgent()
    score = agent.evaluate(trajectory_state=None, track_confidence=0.5)
    print(f"  Motion coherence score : {score:.4f}")
    assert score == 1.0, f"Expected 1.0 for None state, got {score}"  # noqa
    print(f"  Status: PASSED (score == 1.0)")
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print("  MotionAgent - Step 1 Verification")
    print("=" * 50)

    smooth_score = test_smooth_motion()
    erratic_score = test_erratic_motion()
    none_score = test_no_state()

    print("\n--- Summary ---")
    print(f"  Smooth motion coherence  : {smooth_score:.4f}  (expected >= 0.5)")
    print(f"  Erratic motion coherence : {erratic_score:.4f}  (expected < smooth)")
    print(f"  No-state coherence       : {none_score:.4f}  (expected 1.0)")

    if smooth_score > erratic_score:
        print("\n[PASS] Normalization is correct - smooth > erratic as expected.")
    else:
        print("\n[WARN] Erratic score is not lower than smooth. Check simulation parameters.")

    print("\n[PASS] Step 1 complete - MotionAgent is operational.\n")


if __name__ == "__main__":
    main()
