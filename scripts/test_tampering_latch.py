
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.tracking.track import Track
from coherex.tracking.states import TrackState

class TestTamperingLatch(unittest.TestCase):
    def test_latch_tampering(self):
        # 1. Setup Track
        start_frame = 10
        track = Track(track_id=1, init_position=(100, 100), frame_id=start_frame)
        
        # Verify initial state
        self.assertEqual(track.tamper_until_frame, -1, "Initial tamper frame should be -1")
        
        # 2. Trigger Latch
        current_frame = 20
        window = 10
        track.latch_tampering(current_frame, window)
        
        expected_tamper_end = current_frame + window # 30
        self.assertEqual(track.tamper_until_frame, expected_tamper_end, 
                         f"Tamper frame should be {expected_tamper_end}")
        
        # 3. Simulate frames checking against latch
        
        # Case A: Inside Window
        test_frame_inside = 25
        is_tampered_inside = test_frame_inside <= track.tamper_until_frame
        self.assertTrue(is_tampered_inside, "Frame 25 should be considered tampered")
        
        # Case B: On Boundary
        test_frame_boundary = 30
        is_tampered_boundary = test_frame_boundary <= track.tamper_until_frame
        self.assertTrue(is_tampered_boundary, "Frame 30 should be considered tampered")
        
        # Case C: Outside Window
        test_frame_outside = 31
        is_tampered_outside = test_frame_outside <= track.tamper_until_frame
        self.assertFalse(is_tampered_outside, "Frame 31 should NOT be considered tampered")

    def test_latch_extension(self):
        """Ensure latch extends if triggered again with later frame"""
        track = Track(track_id=2, init_position=(100,100), frame_id=0)
        
        # First trigger
        track.latch_tampering(10, 10) # Ends at 20
        self.assertEqual(track.tamper_until_frame, 20)
        
        # Shorter trigger (should be ignored if it ends earlier/same, 
        # but logic is max(current, new_end). 
        # max(20, 15+2=17) -> 20. 
        track.latch_tampering(15, 2) 
        self.assertEqual(track.tamper_until_frame, 20)

        # Extension trigger
        track.latch_tampering(15, 10) # Ends at 25
        self.assertEqual(track.tamper_until_frame, 25)

if __name__ == '__main__':
    unittest.main()
