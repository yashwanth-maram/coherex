"""
scripts/test_segment_aggregator.py

Unit test for SegmentIntegrityAggregator (Step 5 verification).

Tests:
  1. Stable segment (all 0.95)        -> mean ~0.95, volatility ~0.0
  2. Post-anomaly burst (0.2, 0.3)    -> mean drops, min = 0.2, volatility rises
  3. Full window eviction             -> window size is respected
  4. Empty aggregator defaults        -> segment=1.0, min=1.0, volatility=0.0
  5. Snapshot dict structure          -> all keys present and valid

Run with:
    python scripts/test_segment_aggregator.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.segment_aggregator import SegmentIntegrityAggregator


def run_test(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label:<40}{suffix}")
    return condition


def main():
    print("=" * 65)
    print("  SegmentIntegrityAggregator - Step 5 Verification")
    print("=" * 65)

    results = []

    # ------------------------------------------------------------------
    # Test 1: Empty aggregator defaults
    # ------------------------------------------------------------------
    agg_empty = SegmentIntegrityAggregator(window_size=10)
    results.append(run_test("Empty: segment_score = 1.0",
                            agg_empty.get_segment_score() == 1.0))
    results.append(run_test("Empty: min_score = 1.0",
                            agg_empty.get_min_score() == 1.0))
    results.append(run_test("Empty: volatility = 0.0",
                            agg_empty.get_volatility() == 0.0))

    # ------------------------------------------------------------------
    # Test 2: Stable segment — 10 frames of 0.95
    # ------------------------------------------------------------------
    agg = SegmentIntegrityAggregator(window_size=10)
    for _ in range(10):
        agg.update(0.95)

    seg = agg.get_segment_score()
    vol = agg.get_volatility()
    results.append(run_test("Stable: mean ~ 0.95",
                            abs(seg - 0.95) < 1e-9,
                            f"got {seg:.6f}"))
    results.append(run_test("Stable: volatility ~ 0.0",
                            vol < 1e-9,
                            f"got {vol:.8f}"))

    print()
    print("  After 10x stable frames (0.95):")
    print(f"    segment_score : {seg:.4f}")
    print(f"    min_score     : {agg.get_min_score():.4f}")
    print(f"    volatility    : {vol:.6f}")

    # ------------------------------------------------------------------
    # Test 3: Inject anomaly burst (2 low-integrity frames)
    # Window is 10 — these replace the oldest stable frames
    # ------------------------------------------------------------------
    agg.update(0.2)
    agg.update(0.3)

    seg2 = agg.get_segment_score()
    min2 = agg.get_min_score()
    vol2 = agg.get_volatility()

    results.append(run_test("Post-anomaly: mean < stable mean",
                            seg2 < seg,
                            f"got {seg2:.4f}"))
    results.append(run_test("Post-anomaly: min_score = 0.2",
                            abs(min2 - 0.2) < 1e-9,
                            f"got {min2:.4f}"))
    results.append(run_test("Post-anomaly: volatility > 0",
                            vol2 > 0,
                            f"got {vol2:.4f}"))

    print()
    print("  After injecting anomaly frames (0.2, 0.3):")
    print(f"    segment_score : {seg2:.4f}")
    print(f"    min_score     : {min2:.4f}")
    print(f"    volatility    : {vol2:.4f}")

    # ------------------------------------------------------------------
    # Test 4: Window size is enforced (maxlen eviction)
    # ------------------------------------------------------------------
    agg_small = SegmentIntegrityAggregator(window_size=5)
    for v in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:   # 6 values into size-5 window
        agg_small.update(v)

    results.append(run_test("Window eviction: len(scores) == 5",
                            len(agg_small.scores) == 5,
                            f"got {len(agg_small.scores)}"))
    results.append(run_test("Window eviction: oldest frame evicted (1.0 gone)",
                            1.0 not in agg_small.scores,
                            "1.0 should have been evicted"))

    # ------------------------------------------------------------------
    # Test 5: Snapshot dict structure
    # ------------------------------------------------------------------
    snap = agg.get_snapshot()
    expected_keys = {"segment_score", "min_score", "volatility", "frames_in_window"}
    results.append(run_test("Snapshot: all keys present",
                            expected_keys == set(snap.keys()),
                            str(snap.keys())))
    results.append(run_test("Snapshot: frames_in_window == 10",
                            snap["frames_in_window"] == 10,
                            f"got {snap['frames_in_window']}"))

    print()
    print("  Snapshot output:", snap)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(results)
    total = len(results)
    print()
    if passed == total:
        print(f"[PASS] All {total}/{total} tests passed. SegmentAggregator is operational.")
    else:
        print(f"[WARN] {passed}/{total} tests passed.")
    print()


if __name__ == "__main__":
    main()
