"""
scripts/test_crowd_agent.py

Unit test for CrowdAgent (Step 3 verification).

Tests five scenarios:
  1. All tracks moving in the same direction -> expected: ~1.0
  2. Tracks perfectly diverging (opposing)  -> expected: low (~0.0)
  3. Mixed crowd (semi-coherent)             -> expected: mid-range
  4. Only 1 moving track (below min_tracks) -> expected: 1.0 (no judgment)
  5. All tracks stationary (filtered out)   -> expected: 1.0 (no judgment)

Run with:
    python scripts/test_crowd_agent.py
"""

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.crowd_agent import CrowdAgent


# ---------------------------------------------------------------------------
# Minimal track stub (no Kalman needed for this test)
# ---------------------------------------------------------------------------

class StubTrack:
    """Lightweight stand-in for a real Track — only exposes vx, vy."""
    def __init__(self, vx: float, vy: float):
        self.vx = vx
        self.vy = vy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_test(label: str, score: float, expected_min: float, expected_max: float):
    status = "PASS" if expected_min <= score <= expected_max else "FAIL"
    print(f"  [{status}] {label:<40} score={score:.4f}  "
          f"(expected {expected_min:.2f} - {expected_max:.2f})")
    return status == "PASS"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def main():
    agent = CrowdAgent(min_speed_threshold=0.5, min_tracks=2, max_deviation=1.2)

    print("=" * 65)
    print("  CrowdAgent - Step 3 Verification")
    print("=" * 65)

    results = []

    # ------------------------------------------------------------------
    # Test 1: Perfect coherence — all tracks moving right
    # ------------------------------------------------------------------
    tracks_coherent = [StubTrack(5.0, 0.0) for _ in range(6)]
    s1 = agent.evaluate(tracks_coherent)
    results.append(run_test("All moving right (coherent)", s1, 0.95, 1.0))

    # ------------------------------------------------------------------
    # Test 2: Full divergence — half move right, half move left
    # ------------------------------------------------------------------
    tracks_divergent = (
        [StubTrack( 5.0,  0.0) for _ in range(4)] +
        [StubTrack(-5.0,  0.0) for _ in range(4)]
    )
    s2 = agent.evaluate(tracks_divergent)
    results.append(run_test("Opposing halves (full divergence)", s2, 0.0, 0.4))

    # ------------------------------------------------------------------
    # Test 3: Semi-coherent — main direction + one rogue track
    # ------------------------------------------------------------------
    tracks_mixed = (
        [StubTrack(4.0, 1.0) for _ in range(5)] +    # rightward crowd
        [StubTrack(0.0, 5.0)]                         # one going up
    )
    s3 = agent.evaluate(tracks_mixed)
    results.append(run_test("5 right + 1 up (semi-coherent)", s3, 0.3, 0.85))

    # ------------------------------------------------------------------
    # Test 4: Radial divergence — tracks spread in all directions
    # ------------------------------------------------------------------
    angles = [i * (360 / 8) for i in range(8)]
    tracks_radial = [
        StubTrack(3.0 * math.cos(math.radians(a)), 3.0 * math.sin(math.radians(a)))
        for a in angles
    ]
    s4 = agent.evaluate(tracks_radial)
    results.append(run_test("8 radial directions (chaos)", s4, 0.0, 0.35))

    # ------------------------------------------------------------------
    # Test 5: Only 1 moving track — cannot judge crowd
    # ------------------------------------------------------------------
    tracks_single = [StubTrack(5.0, 0.0)]
    s5 = agent.evaluate(tracks_single)
    results.append(run_test("Single moving track (no judgment)", s5, 1.0, 1.0))

    # ------------------------------------------------------------------
    # Test 6: All tracks stationary — filtered out, cannot judge
    # ------------------------------------------------------------------
    tracks_stationary = [StubTrack(0.0, 0.0) for _ in range(10)]
    s6 = agent.evaluate(tracks_stationary)
    results.append(run_test("All stationary (filtered)", s6, 1.0, 1.0))

    # ------------------------------------------------------------------
    # Test 7: Empty track list
    # ------------------------------------------------------------------
    s7 = agent.evaluate([])
    results.append(run_test("Empty track list", s7, 1.0, 1.0))

    print()
    print("--- Summary ---")
    print(f"  All coherent          : {s1:.4f}")
    print(f"  Full divergence       : {s2:.4f}")
    print(f"  Semi-coherent (rogue) : {s3:.4f}")
    print(f"  Radial chaos          : {s4:.4f}")
    print(f"  Single track          : {s5:.4f}")
    print(f"  All stationary        : {s6:.4f}")
    print(f"  Empty list            : {s7:.4f}")

    passed = sum(results)
    total = len(results)
    print()
    if passed == total:
        print(f"[PASS] All {total}/{total} tests passed. CrowdAgent is operational.")
    else:
        print(f"[WARN] {passed}/{total} tests passed. Review failing cases.")
    print()


if __name__ == "__main__":
    main()
