import numpy as np
from .states import TrackState
from .kalman import create_kalman_filter


class Track:
    """
    Represents a single object track across time.
    Handles identity, lifecycle, and Kalman-based motion estimation.
    """

    def __init__(self, track_id, init_position, frame_id):
        self.track_id = track_id
        self.state = TrackState.NEW

        # Last known appearance proxy
        self.last_bbox_area = None
        self.last_bbox_ratio = None

        # Kalman Filter
        self.kf = create_kalman_filter(init_position[0], init_position[1])
        self.covariance = self.kf.P

        # State variables
        self.x = init_position[0]
        self.y = init_position[1]
        self.vx = 0.0
        self.vy = 0.0

        # Lifecycle stats
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.reattach_count = 0  # times this track was re-attached after dormancy

        self.first_seen_frame = frame_id
        self.last_seen_frame = frame_id

        self.confidence = 0.1

        # Tampering event latch (frame-based)
        self.tamper_until_frame = -1
        
        self.color_hist = None
        
        self.pose_scale = None
        
        # Identity Structure History (new)
        self.structure_history = []
        self.max_structure_history = 20

    # -----------------------------
    # Kalman motion methods
    # -----------------------------

    def predict(self):
        self.kf.predict()
        self._sync_from_kalman()

    def update(self, meas_x, meas_y):
        measurement = np.array([[meas_x], [meas_y]])
        self.kf.update(measurement)
        self._sync_from_kalman()

    def _sync_from_kalman(self):
        self.x, self.y, self.vx, self.vy = self.kf.x.flatten()
        self.covariance = self.kf.P

    # -----------------------------
    # Lifecycle methods
    # -----------------------------

    def mark_hit(self, frame_id):
        self.hits += 1
        self.misses = 0
        self.last_seen_frame = frame_id
        self.age += 1

        self.confidence = min(1.0, self.confidence + 0.1)

        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.ACTIVE
        elif self.state == TrackState.DORMANT:
            self.state = TrackState.ACTIVE

    def mark_miss(self):
        self.misses += 1
        self.age += 1

        if self.state == TrackState.ACTIVE:
            self.state = TrackState.DORMANT

    def should_terminate(self, max_misses):
        return self.misses > max_misses

    def terminate(self):
        self.state = TrackState.TERMINATED

    def update_appearance(self, bbox):
        """
        Store lightweight appearance proxy from bbox.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            self.last_bbox_area = w * h
            self.last_bbox_ratio = w / h

    def update_appearance(self, hist):
        self.color_hist = hist

    def update_pose_scale(self, scale):
        if scale is not None:
            self.pose_scale = scale

    def update_structure(self, structure):
        """
        Update body structure history.
        structure: BodyStructure object
        """
        if structure is None:
            return
            
        self.structure_history.append(structure)
        if len(self.structure_history) > self.max_structure_history:
            self.structure_history.pop(0)

    def latch_tampering(self, current_frame, window_frames):
        """
        Escalate identity break to a tampering event
        for a fixed temporal window.
        """
        self.tamper_until_frame = max(
            self.tamper_until_frame,
            current_frame + window_frames
        )

