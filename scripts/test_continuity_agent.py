"""
scripts/test_continuity_agent.py

Unit test for ContinuityAgent (Step 2 verification).

Tests five scenarios:
  1. Stable track              -> expected: 1.0
  2. Miss-heavy track (5)      -> expected: 0.5
  3. Dormant track (2 misses)  -> expected: 0.6
  4. Tampered track            -> expected: 0.0
  5. Multi-reattach track      -> expected: < stable
  6. Terminated track          -> expected: 0.0

Run with:
    python scripts/test_continuity_agent.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.continuity_agent import ContinuityAgent
from coherex.tracking.track import Track
from coherex.tracking.states import TrackState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(
    misses: int = 0,
    state: TrackState = TrackState.ACTIVE,
    tamper_until: int = -1,
    reattach_count: int = 0,
) -> Track:
    """Build a Track object with the given lifecycle properties."""
    track = Track(track_id=0, init_position=(0, 0), frame_id=0)
    track.misses = misses
    track.state = state
    track.tamper_until_frame = tamper_until
    track.reattach_count = reattach_count
    # Promote to ACTIVE so state.name reads correctly
    return track


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def run_test(label: str, score: float, expected_min: float, expected_max: float):
    status = "PASS" if expected_min <= score <= expected_max else "FAIL"
    print(f"  [{status}] {label:<30} score={score:.4f}  (expected {expected_min:.1f} - {expected_max:.1f})")
    return status == "PASS"


def main():
    agent = ContinuityAgent(
        miss_penalty_scale=10.0,
        dormant_penalty=0.2,
        reattach_penalty=0.15,
        max_reattach_penalty=0.45,
    )

    print("=" * 60)
    print("  ContinuityAgent - Step 2 Verification")
    print("=" * 60)

    results = []

    # --- Test 1: Stable, active, no misses, no reattachments ---
    t1 = make_track(misses=0, state=TrackState.ACTIVE)
    s1 = agent.evaluate(t1, frame_id=20)
    results.append(run_test("Stable track", s1, 1.0, 1.0))

    # --- Test 2: Miss-heavy (5 out of 10 threshold = -0.5) ---
    t2 = make_track(misses=5, state=TrackState.ACTIVE)
    s2 = agent.evaluate(t2, frame_id=20)
    results.append(run_test("Miss-heavy (5 misses)", s2, 0.4, 0.6))

    # --- Test 3: Dormant + 2 misses = -0.2 - 0.2 = 0.6 ---
    t3 = make_track(misses=2, state=TrackState.DORMANT)
    s3 = agent.evaluate(t3, frame_id=20)
    results.append(run_test("Dormant (2 misses)", s3, 0.5, 0.7))

    # --- Test 4: Tamper latch active (frame_id=5, tamper_until=10) ---
    t4 = make_track(tamper_until=10)
    s4 = agent.evaluate(t4, frame_id=5)
    results.append(run_test("Tampered (latch active)", s4, 0.0, 0.0))

    # --- Test 5: Tamper latch expired (frame_id=15, tamper_until=10) ---
    t5 = make_track(tamper_until=10)
    s5 = agent.evaluate(t5, frame_id=15)
    results.append(run_test("Tamper latch expired", s5, 0.9, 1.0))

    # --- Test 6: Re-attached 2x = -0.30 ---
    t6 = make_track(reattach_count=2)
    s6 = agent.evaluate(t6, frame_id=20)
    results.append(run_test("2x reattached track", s6, 0.6, 0.8))

    # --- Test 7: Re-attached 4x (capped at max_reattach_penalty=0.45) ---
    t7 = make_track(reattach_count=4)
    s7 = agent.evaluate(t7, frame_id=20)
    results.append(run_test("4x reattached (cap)", s7, 0.5, 0.6))

    # --- Test 8: Terminated track ---
    t8 = make_track(state=TrackState.TERMINATED)
    s8 = agent.evaluate(t8, frame_id=20)
    results.append(run_test("Terminated track", s8, 0.0, 0.0))

    print()
    print("--- Summary ---")
    print(f"  Stable              : {s1:.4f}")
    print(f"  Miss-heavy (5)      : {s2:.4f}  (penalty = {1.0-s2:.4f})")
    print(f"  Dormant + 2 misses  : {s3:.4f}  (penalty = {1.0-s3:.4f})")
    print(f"  Tampered (active)   : {s4:.4f}")
    print(f"  Tamper (expired)    : {s5:.4f}")
    print(f"  Reattach x2         : {s6:.4f}  (penalty = {1.0-s6:.4f})")
    print(f"  Reattach x4 (cap)   : {s7:.4f}  (penalty = {1.0-s7:.4f})")
    print(f"  Terminated          : {s8:.4f}")

    passed = sum(results)
    total = len(results)
    print()
    if passed == total:
        print(f"[PASS] All {total}/{total} tests passed. ContinuityAgent is operational.")
    else:
        print(f"[WARN] {passed}/{total} tests passed. Review failing cases.")

    print()


if __name__ == "__main__":
    main()
