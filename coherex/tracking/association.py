import numpy as np
from scipy.optimize import linear_sum_assignment


def associate_tracks_to_detections(tracks, detections, max_distance=80):
    """
    Associate tracks with detections using Hungarian algorithm.

    Args:
        tracks (list[Track]): existing tracks
        detections (list[tuple]): list of (x, y) detection centers
        max_distance (float): distance threshold

    Returns:
        matches (list[tuple]): (track_idx, detection_idx)
        unmatched_tracks (list[int])
        unmatched_detections (list[int])
    """

    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            dx = track.x - det[0]
            dy = track.y - det[1]
            cost_matrix[i, j] = np.sqrt(dx * dx + dy * dy)

    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for t_idx, d_idx in zip(track_indices, detection_indices):
        if cost_matrix[t_idx, d_idx] <= max_distance:
            matches.append((t_idx, d_idx))
            unmatched_tracks.remove(t_idx)
            unmatched_detections.remove(d_idx)

    return matches, unmatched_tracks, unmatched_detections
