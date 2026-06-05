# coherex/trajectory/store.py

from .state import TrajectoryState
from coherex.config import CONFIG


class TrajectoryStore:
    """
    Maintains TrajectoryState objects for active tracks.
    """

    def __init__(self, max_history=None, config=None):
        self.states = {}
        cfg = config or CONFIG
        self.max_history = (
            max_history if max_history is not None
            else cfg.tracking.max_trajectory_history
        )

    def update_from_tracks(self, tracks):
        """
        Update trajectory states from current active tracks.
        """
        for track in tracks:
            if track.track_id not in self.states:
                self.states[track.track_id] = TrajectoryState(
                    track.track_id, self.max_history
                )

            state = self.states[track.track_id]
            state.update(track.x, track.y, track.vx, track.vy)

    def get_state(self, track_id):
        return self.states.get(track_id, None)
