import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.track import Track
from coherex.tracking.association import associate_tracks_to_detections

# Create dummy tracks
tracks = [
    Track(1, (100, 100), 0),
    Track(2, (300, 300), 0)
]

# Predict to simulate motion
for t in tracks:
    t.predict()

# Detections near track 1 and a new object
detections = [
    (105, 102),   # near track 1
    (500, 500)    # new object
]

matches, unmatched_tracks, unmatched_detections = associate_tracks_to_detections(
    tracks, detections
)

print("Matches:", matches)
print("Unmatched Tracks:", unmatched_tracks)
print("Unmatched Detections:", unmatched_detections)
