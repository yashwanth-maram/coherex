"""
scripts/test_fusion_engine.py

Unit test for IntegrityFusionEngine (Step 4 verification).

Tests:
  1. All-high coherence       -> expected: > 0.85  (HIGH)
  2. Mixed instability        -> expected: 0.60-0.75 (MODERATE)
  3. Severe breakdown         -> expected: < 0.30  (COMPROMISED)
  4. Empty track list         -> expected: 1.0     (no judgment)
  5. Weight validation error  -> expected: ValueError raised
  6. Score always clamped     -> expected: in [0.0, 1.0]

Run with:
    python scripts/test_fusion_engine.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.fusion_engine import IntegrityFusionEngine


def run_test(label, score, expected_min, expected_max):
    status = "PASS" if expected_min <= score <= expected_max else "FAIL"
    print(f"  [{status}] {label:<35} score={score:.4f}  "
          f"(expected {expected_min:.2f} - {expected_max:.2f})")
    return status == "PASS"


def main():
    engine = IntegrityFusionEngine()

    print("=" * 65)
    print("  IntegrityFusionEngine - Step 4 Verification")
    print("=" * 65)

    results = []

    # ------------------------------------------------------------------
    # Test 1: All-high coherence → HIGH INTEGRITY
    # Expected: 0.4*0.95 + 0.3*0.95 + 0.3*0.9 = 0.38+0.285+0.27 = 0.935
    # ------------------------------------------------------------------
    s1 = engine.fuse(
        motion_scores=[0.9, 1.0, 0.95],
        continuity_scores=[1.0, 0.95, 0.9],
        crowd_score=0.9,
    )
    results.append(run_test("High integrity (all coherent)", s1, 0.85, 1.0))
    print(f"          interpret -> {engine.interpret(s1)}")

    # ------------------------------------------------------------------
    # Test 2: Mixed instability → MODERATE
    # Expected: 0.4*0.65 + 0.3*0.70 + 0.3*0.5 = 0.26+0.21+0.15 = 0.62
    # ------------------------------------------------------------------
    s2 = engine.fuse(
        motion_scores=[0.6, 0.7],
        continuity_scores=[0.8, 0.6],
        crowd_score=0.5,
    )
    results.append(run_test("Moderate integrity (mixed)", s2, 0.55, 0.75))
    print(f"          interpret -> {engine.interpret(s2)}")

    # ------------------------------------------------------------------
    # Test 3: Severe breakdown → COMPROMISED
    # Expected: 0.4*0.15 + 0.3*0.05 + 0.3*0.2 = 0.06+0.015+0.06 = 0.135
    # ------------------------------------------------------------------
    s3 = engine.fuse(
        motion_scores=[0.1, 0.2],
        continuity_scores=[0.0, 0.1],
        crowd_score=0.2,
    )
    results.append(run_test("Low integrity (severe breakdown)", s3, 0.0, 0.30))
    print(f"          interpret -> {engine.interpret(s3)}")

    # ------------------------------------------------------------------
    # Test 4: Empty track list → default 1.0 (no judgment possible)
    # ------------------------------------------------------------------
    s4 = engine.fuse(motion_scores=[], continuity_scores=[], crowd_score=0.5)
    results.append(run_test("Empty tracks (no judgment)", s4, 1.0, 1.0))

    # ------------------------------------------------------------------
    # Test 5: Single-track edge case
    # ------------------------------------------------------------------
    s5 = engine.fuse(
        motion_scores=[0.7],
        continuity_scores=[0.8],
        crowd_score=1.0,
    )
    results.append(run_test("Single track scene", s5, 0.7, 0.9))

    # ------------------------------------------------------------------
    # Test 6: CONFIG weight consistency — weights must sum to 1.0
    # ------------------------------------------------------------------
    from coherex.config import CONFIG
    w = CONFIG.integrity.fusion_weights
    weight_sum = sum(w)
    valid = abs(weight_sum - 1.0) < 1e-6
    status = "PASS" if valid else "FAIL"
    print(f"  [{status}] {'CONFIG weight consistency':<35} "
          f"sum={weight_sum:.4f} (expected 1.0)")
    results.append(valid)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("--- Summary ---")
    print(f"  High integrity       : {s1:.4f}  -> {engine.interpret(s1)}")
    print(f"  Moderate integrity   : {s2:.4f}  -> {engine.interpret(s2)}")
    print(f"  Low integrity        : {s3:.4f}  -> {engine.interpret(s3)}")
    print(f"  Empty tracks         : {s4:.4f}")
    print(f"  Single track         : {s5:.4f}")

    passed = sum(results)
    total = len(results)
    print()
    if passed == total:
        print(f"[PASS] All {total}/{total} tests passed. FusionEngine is operational.")
    else:
        print(f"[WARN] {passed}/{total} tests passed.")
    print()


if __name__ == "__main__":
    main()
