import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore

tm = TrackManager()
store = TrajectoryStore()

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
tm.current_frame = dummy_frame
tm.current_pose = {}

# Simulated motion
for f in range(10):
    cx, cy = 100 + f * 5, 100
    detections = [(cx, cy)]
    bboxes = [(cx - 20, cy - 50, cx + 20, cy + 50)]
    tracks = tm.update(detections=detections, bboxes=bboxes, frame_id=f)
    store.update_from_tracks(tracks)

state = store.get_state(0)

print("Speeds:", list(state.speeds))
print("Accelerations:", list(state.accelerations))
print("Angles:", list(state.angles))
