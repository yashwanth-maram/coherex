import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.track import Track

t = Track(track_id=1, init_position=(100, 100), frame_id=0)

print("Initial:", t.x, t.y)

# Simulate motion with detections
for i in range(1, 6):
    t.predict()
    t.update(100 + i * 5, 100)
    print(f"Frame {i}: pos=({t.x:.2f}, {t.y:.2f}), vel=({t.vx:.2f}, {t.vy:.2f})")

# Simulate occlusion (no detection)
for i in range(6, 9):
    t.predict()
    print(f"Frame {i} (no detection): pos=({t.x:.2f}, {t.y:.2f})")
