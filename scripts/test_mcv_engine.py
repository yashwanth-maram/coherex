import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore
from coherex.coherence.mcv import MotionCoherenceEngine

tm = TrackManager()
store = TrajectoryStore()
mcv_engine = MotionCoherenceEngine()

# Provide a dummy frame for appearance extraction
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
tm.current_frame = dummy_frame
tm.current_pose = {}

# Simulate mostly smooth motion, then a sudden jump
for f in range(15):
    if f < 10:
        cx, cy = 100 + f * 5, 100
    else:
        cx, cy = 200, 300  # sudden jump

    detections = [(cx, cy)]
    bboxes = [(cx - 20, cy - 50, cx + 20, cy + 50)]

    tracks = tm.update(detections=detections, bboxes=bboxes, frame_id=f)
    store.update_from_tracks(tracks)

    for t in tracks:
        state = store.get_state(t.track_id)
        mcv = mcv_engine.compute(state, t.confidence)
        print(f"Frame {f}, Track {t.track_id}, MCV: {mcv:.3f}")
