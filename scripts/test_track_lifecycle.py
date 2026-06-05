import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.track import Track
from coherex.tracking.states import TrackState

t = Track(track_id=1, init_position=(100, 200), frame_id=0)

print(t.state)  # NEW

t.mark_hit(1)
t.mark_hit(2)
t.mark_hit(3)

print(t.state)  # ACTIVE

t.mark_miss()
print(t.state)  # DORMANT

t.mark_hit(5)
print(t.state)  # ACTIVE
