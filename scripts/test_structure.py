import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.pose import compute_body_structure, BodyStructure
# from coherex.tracking.track import Track
from coherex.tracking.identity import body_structure_consistent

class MockTrack:
    def __init__(self):
        self.track_id = 1
        self.structure_history = []
        self.max_structure_history = 20
        self.pose_scale = None # legacy

    def update_structure(self, structure):
        if structure is None: return
        self.structure_history.append(structure)
        if len(self.structure_history) > self.max_structure_history:
            self.structure_history.pop(0)

def generate_mock_keypoints(scale_factor=1.0, noise=0.0):
    """
    Generate a standard human pose scaled by scale_factor.
    17 keypoints: [x, y, conf]
    """
    # Base height ~ 170 units (pixels or cm)
    # Head at (0, 0)
    # Hip at (0, 80)
    # Feet at (0, 170)
    # Shoulders at (-15, 20), (15, 20)
    
    # Indices:
    # 0: Nose
    # 5, 6: Shoulders
    # 11, 12: Hips
    # 15, 16: Ankles
    
    kp = np.zeros((17, 3))
    
    # Nose
    kp[0] = [0, 0, 1]
    
    # Shoulders
    kp[5] = [-20 * scale_factor, 20 * scale_factor, 1]
    kp[6] = [ 20 * scale_factor, 20 * scale_factor, 1]
    
    # Hips
    kp[11] = [-15 * scale_factor, 80 * scale_factor, 1]
    kp[12] = [ 15 * scale_factor, 80 * scale_factor, 1]
    
    # Ankles
    kp[15] = [-15 * scale_factor, 170 * scale_factor, 1]
    kp[16] = [ 15 * scale_factor, 170 * scale_factor, 1]
    
    # Add noise
    if noise > 0:
        kp[:, :2] += np.random.normal(0, noise, size=(17, 2))
        
    return kp

def test_structure_invariant_to_scale():
    print("\n--- Test: Scale Invariance ---")
    kp_adult = generate_mock_keypoints(scale_factor=1.0)
    kp_zoomed_adult = generate_mock_keypoints(scale_factor=1.5) # Same person, closer to camera
    
    s1 = compute_body_structure(kp_adult)
    s2 = compute_body_structure(kp_zoomed_adult)
    
    print(f"Adult Features: {s1.vector}")
    print(f"Zoomed Features: {s2.vector}")
    
    dist = np.linalg.norm(s1.vector - s2.vector)
    print(f"Distance (should be small): {dist:.4f}")
    assert dist < 0.1, "Structure should be scale invariant"

def test_structure_sensitive_to_proportion():
    print("\n--- Test: Proportion Sensitivity (Adult vs Child) ---")
    # Adult
    kp_adult = generate_mock_keypoints(scale_factor=1.0)
    
    # Child: Shorter legs relative to torso
    # Let's manually modify legs to be shorter
    kp_child = generate_mock_keypoints(scale_factor=0.6) # Smaller overall
    # BUT children have different ratios. Head is larger, legs are shorter relative to torso.
    # Torso length (Shoulder-Hip) is approx 60 units in base. 
    # Leg length (Hip-Ankle) is approx 90 units in base. Ratio ~ 1.5
    
    # Modify child legs to be shorter relative to torso -> Ratio ~ 1.0
    # Keep torso same as scaled child, shorten legs.
    # Current scaled child hips at 80*0.6 = 48. Ankles at 170*0.6 = 102. Leg len = 54. Torso len = 36. Ratio 1.5.
    # Let's move ankles up to make legs shorter.
    kp_child[15][1] = kp_child[11][1] + 36 # Leg len = Torso len
    kp_child[16][1] = kp_child[12][1] + 36
    
    s_adult = compute_body_structure(kp_adult)
    s_child = compute_body_structure(kp_child)
    
    print(f"Adult Features: {s_adult.vector}")
    print(f"Child Features: {s_child.vector}")
    
    dist = np.linalg.norm(s_adult.vector - s_child.vector)
    print(f"Distance (should be large): {dist:.4f}")
    assert dist > 0.4, "Structure should be different for different proportions"

def test_consistency_check():
    print("\n--- Test: Consistency Check Over Time ---")
    track = MockTrack()
    
    # 1. Warm up track with Adult structure (with some noise)
    print("Warming up track with 20 frames of Adult...")
    for _ in range(20):
        kp = generate_mock_keypoints(scale_factor=1.0, noise=2.0)
        s = compute_body_structure(kp)
        track.update_structure(s)
        
    # 2. Check consistency with another Adult frame
    kp_new_adult = generate_mock_keypoints(scale_factor=1.0, noise=2.0)
    s_new_adult = compute_body_structure(kp_new_adult)
    
    consistent = body_structure_consistent(track, s_new_adult)
    print(f"New Adult Consistent? {consistent}")
    assert consistent is True
    
    # 3. Check consistency with Child frame (Tampering Attempt)
    # Using the modified child proportions from before
    kp_child = generate_mock_keypoints(scale_factor=1.0) # Same scale, just to isolate proportion
    # Modifying legs
    kp_child[15][1] = kp_child[11][1] + 36 # Leg len = Torso len (approx 1.0 ratio, vs 1.5 adult)
    kp_child[16][1] = kp_child[12][1] + 36
    
    s_child = compute_body_structure(kp_child)
    consistent_child = body_structure_consistent(track, s_child)
    print(f"Child Swap Consistent? {consistent_child}")
    assert consistent_child is False

if __name__ == "__main__":
    test_structure_invariant_to_scale()
    test_structure_sensitive_to_proportion()
    test_consistency_check()
    print("\nALL SCENARIOS PASSED")
