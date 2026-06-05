"""
scripts/test_dynamic_fusion.py

Verifies Step 8: Agent Reliability Estimation (Dynamic Fusion).

Scenarios:
  1. Mature scene (all agents reliable) -> similar to static weights.
  2. Single Person Scene -> Crowd reliability drops -> Crowd weight collapses.
  3. New/Unstable tracks -> Motion reliability drops -> Motion weight reduces.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.integrity.fusion_engine import IntegrityFusionEngine
from coherex.integrity.reliability import ReliabilityEstimator

class MockTrack:
    def __init__(self, track_id, age, confidence):
        self.track_id = track_id
        self.age = age
        self.confidence = confidence

class MockTrajectoryState:
    def __init__(self, history_len):
        self.speeds = [1.0] * history_len

class MockTrajectoryStore:
    def __init__(self, states_map):
        self.states_map = states_map
    def get_state(self, track_id):
        return self.states_map.get(track_id)

def main():
    rel_est = ReliabilityEstimator()
    # Base weights: 0.4, 0.3, 0.3 (static)
    engine = IntegrityFusionEngine(reliability_estimator=rel_est)

    print("=" * 65)
    print("  Reliability-Aware Dynamic Fusion Verification")
    print("=" * 65)

    # Scenarios use high coherence scores for the math to be obvious
    ms = [1.0] # motion score
    cs = [1.0] # continuity score
    crs = 0.5  # crowd score (lowered to see if its weight changes)

    # ------------------------------------------------------------------
    # Scenario 1: Mature / Reliable Scene
    # 5 tracks, all age 15, all high confidence, all history full
    # Expected: R_motion=1.0, R_cont=1.0, R_crowd=1.0 -> weights stay 0.4, 0.3, 0.3
    # Result: 0.4*1 + 0.3*1 + 0.3*0.5 = 0.4 + 0.3 + 0.15 = 0.85
    # ------------------------------------------------------------------
    tracks1 = [MockTrack(i, 15, 1.0) for i in range(5)]
    store1 = MockTrajectoryStore({i: MockTrajectoryState(10) for i in range(5)})
    
    score1 = engine.fuse(ms * 5, cs * 5, crs, tracks=tracks1, trajectory_store=store1)
    r1m, r1c, r1cr = engine.get_last_reliabilities()
    
    print(f"Scenario 1: Mature Scene (5 tracks, stable)")
    print(f"  Reliabilities: M={r1m:.2f}, C={r1c:.2f}, Cr={r1cr:.2f}")
    print(f"  Final Score:   {score1:.4f} (Expected ~0.85)")
    assert abs(score1 - 0.85) < 0.01

    # ------------------------------------------------------------------
    # Scenario 2: Single Person Scene (Crowd collapse)
    # 1 track, age 20, high confidence, full history
    # R_motion=1.0, R_cont=1.0, R_crowd=0.2 (1/5)
    # Weights scaled: M:0.4*1.0=0.4, C:0.3*1.0=0.3, Cr:0.3*0.2=0.06
    # Sum: 0.76. Normalized: M:0.4/0.76=0.526, C:0.3/0.76=0.395, Cr:0.06/0.76=0.079
    # Final: 0.526*1 + 0.395*1 + 0.079*0.5 = 0.526 + 0.395 + 0.0395 = 0.9605
    # (Crowd penalty of 0.5 is mostly ignored)
    # ------------------------------------------------------------------
    tracks2 = [MockTrack(0, 20, 1.0)]
    store2 = MockTrajectoryStore({0: MockTrajectoryState(10)})
    
    score2 = engine.fuse(ms, cs, crs, tracks=tracks2, trajectory_store=store2)
    r2m, r2c, r2cr = engine.get_last_reliabilities()
    
    print(f"\nScenario 2: Sparse Scene (1 track)")
    print(f"  Reliabilities: M={r2m:.2f}, C={r2c:.2f}, Cr={r2cr:.2f}")
    print(f"  Final Score:   {score2:.4f} (Expected ~0.96)")
    assert score2 > 0.95

    # ------------------------------------------------------------------
    # Scenario 3: Unstable / New tracks (Motion collapse)
    # 5 tracks, age 1, low confidence 0.2, no history
    # R_motion=0.0 (conf*hist=0.2*0=0), R_cont=0.1 (1/10), R_crowd=1.0 (5/5)
    # Weights scaled: M:0.4*0=0, C:0.3*0.1=0.03, Cr:0.3*1.0=0.3
    # Sum: 0.33. Normalized: M:0, C:0.03/0.33=0.09, Cr:0.3/0.33=0.91
    # Final: 0*1 + 0.09*1 + 0.91*0.5 = 0.09 + 0.455 = 0.545
    # (Crowd score 0.5 dominates because it's the only reliable one)
    # ------------------------------------------------------------------
    tracks3 = [MockTrack(i, 1, 0.2) for i in range(5)]
    store3 = MockTrajectoryStore({i: MockTrajectoryState(0) for i in range(5)})
    
    score3 = engine.fuse(ms * 5, cs * 5, crs, tracks=tracks3, trajectory_store=store3)
    r3m, r3c, r3cr = engine.get_last_reliabilities()
    
    print(f"\nScenario 3: Unstable Scene (New tracks)")
    print(f"  Reliabilities: M={r3m:.2f}, C={r3c:.2f}, Cr={r3cr:.2f}")
    print(f"  Final Score:   {score3:.4f} (Expected ~0.55)")
    assert abs(score3 - 0.55) < 0.05

    print("\n[PASS] Dynamic Fusion Logic Verified.")

if __name__ == "__main__":
    main()
