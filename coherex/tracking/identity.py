import numpy as np
from coherex.config import CONFIG


def identity_consistent(track, new_bbox,
                        area_thresh=None,
                        ratio_thresh=None):
    """
    Check whether re-attaching this detection
    preserves identity consistency.
    """
    tc = CONFIG.tracking
    area_thresh = area_thresh if area_thresh is not None else tc.identity_area_range
    ratio_thresh = ratio_thresh if ratio_thresh is not None else tc.identity_ratio_threshold

    if track.last_bbox_area is None:
        return True  # no reference yet

    x1, y1, x2, y2 = new_bbox
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    new_area = w * h
    new_ratio = w / h

    area_ratio = new_area / track.last_bbox_area
    ratio_diff = abs(new_ratio - track.last_bbox_ratio)

    if not (area_thresh[0] <= area_ratio <= area_thresh[1]):
        return False

    if ratio_diff > ratio_thresh:
        return False

    return True


def body_structure_consistent(track, new_structure, sigma_thresh=3.0):
    """
    Check if the new body structure is statistically consistent 
    with the track's history.
    
    Returns: True/False
    """
    if not track.structure_history or new_structure is None:
        return True
        
    history_vecs = np.array([s.vector for s in track.structure_history])
    
    mean_vec = np.mean(history_vecs, axis=0)
    std_vec = np.std(history_vecs, axis=0)
    
    diff = new_structure.vector - mean_vec
    
    epsilon = 1e-3
    std_safe = std_vec + epsilon
    
    z_scores = diff / std_safe
    
    max_z = np.max(np.abs(z_scores))
    
    if max_z > sigma_thresh:
        return False
        
    return True
