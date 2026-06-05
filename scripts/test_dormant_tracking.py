import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.manager import TrackManager

tm = TrackManager(max_misses=2, max_dormant_frames=5)

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
tm.current_frame = dummy_frame
tm.current_pose = {}

# Frame 0–2: person appears
for f in range(3):
    cx, cy = 100 + f * 5, 100
    tracks = tm.update(
        detections=[(cx, cy)],
        bboxes=[(cx - 20, cy - 50, cx + 20, cy + 50)],
        frame_id=f,
    )
    print(f"Frame {f}:", [(t.track_id, t.state.name) for t in tracks])

# Frame 3–4: occlusion (no detections)
for f in range(3, 5):
    tracks = tm.update(detections=[], bboxes=[], frame_id=f)
    print(f"Frame {f}:", [(t.track_id, t.state.name) for t in tracks])

# Frame 5: person reappears nearby
tracks = tm.update(
    detections=[(115, 100)],
    bboxes=[(95, 50, 135, 150)],
    frame_id=5,
)
print("Frame 5:", [(t.track_id, t.state.name) for t in tracks])
