
import numpy as np


class BodyStructure:
    def __init__(self, vector):
        self.vector = vector  # numpy array of features

    def distance_to(self, other):
        return np.linalg.norm(self.vector - other.vector)

    def __repr__(self):
        return f"BodyStructure(dim={len(self.vector)})"


def compute_body_structure(keypoints):
    """
    Compute multi-dimensional body structure features.
    keypoints: (17, 3) or (17, 2) array
    Returns: BodyStructure object or None
    """
    if keypoints is None or len(keypoints) < 17:
        return None

    kp = np.array(keypoints)
    
    # Indices
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_ANKLE, R_ANKLE = 15, 16

    # Points (using only x, y)
    p_nose = kp[NOSE][:2]
    p_l_sh = kp[L_SHOULDER][:2]
    p_r_sh = kp[R_SHOULDER][:2]
    p_l_hip = kp[L_HIP][:2]
    p_r_hip = kp[R_HIP][:2]
    p_l_ank = kp[L_ANKLE][:2]
    p_r_ank = kp[R_ANKLE][:2]

    # Derived points
    p_sh_mid = (p_l_sh + p_r_sh) / 2
    p_hip_mid = (p_l_hip + p_r_hip) / 2
    p_ank_mid = (p_l_ank + p_r_ank) / 2

    # Scale normalization (Torso Length)
    torso_len = np.linalg.norm(p_sh_mid - p_hip_mid)
    if torso_len < 1.0:  # Avoid div/0 for degenerate poses
        return None

    # --- Feature 1: Normalized Distances ---
    # Head to Hip
    d_head_hip = np.linalg.norm(p_nose - p_hip_mid) / torso_len
    
    # Leg Length (Hip to Ankle)
    d_leg = np.linalg.norm(p_hip_mid - p_ank_mid) / torso_len
    
    # Shoulder Width
    d_sh_width = np.linalg.norm(p_l_sh - p_r_sh) / torso_len
    
    # Hip Width
    d_hip_width = np.linalg.norm(p_l_hip - p_r_hip) / torso_len

    # --- Feature 2: Angles (in radians, or cos/sin components) ---
    # Vertical Alignment: Angle between Neck-Hip vector and Hip-Foot vector
    # We'll use cosine of the angle to avoid issues with wrapping, or just raw vectors relative to vertical?
    # Simpler: Angle at Hip (ShoulderMid -> HipMid -> AnkleMid)
    v_torso = p_sh_mid - p_hip_mid
    v_legs = p_ank_mid - p_hip_mid
    
    # Angle 1: Torso-Leg bend (0 = straight, pi = folded)
    # Cosine similarity
    norm_t = np.linalg.norm(v_torso)
    norm_l = np.linalg.norm(v_legs)
    
    if norm_t == 0 or norm_l == 0:
        a_torso_leg = 0.0
    else:
        # dot product
        dot = np.dot(v_torso, v_legs)
        cos_angle = dot / (norm_t * norm_l)
        # We generally expect this to be close to -1 (straight body, vectors pointing away from hip)
        # But wait:
        # v_torso is Hip->Shoulder (up)
        # v_legs is Hip->Ankle (down)
        # So they are roughly opposing. Angle is ~180 deg. Cos is ~ -1.
        a_torso_leg = cos_angle

    # Feature Vector
    features = np.array([
        d_head_hip,
        d_leg,
        d_sh_width,
        d_hip_width,
        a_torso_leg
    ], dtype=np.float32)

    return BodyStructure(features)


def compute_pose_scale(keypoints):
    """
    Legacy wrapper for 1D pose scale.
    """
    struct = compute_body_structure(keypoints)
    if struct is None:
        return None
    # d_leg is index 1
    return struct.vector[1]
