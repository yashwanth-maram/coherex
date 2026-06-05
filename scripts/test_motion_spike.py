# scripts/test_motion_spike.py

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from coherex.trajectory.state import TrajectoryState
from coherex.integrity.motion_agent import MotionAgent
from coherex.config import CONFIG

def run_test():
    output_path = "data/evaluation/spike_test_results.txt"
    os.makedirs("data/evaluation", exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("="*60 + "\n")
        f.write(" MOTION AGENT SPIKE TEST: TELEPORTATION DETECTION\n")
        f.write("="*60 + "\n")
        
        agent = MotionAgent()
        traj = TrajectoryState(track_id=1, max_history=10)
        
        # Mock a Track object with a latching mechanism
        class MockTrack:
            def __init__(self):
                self.tamper_until_frame = -1
            def latch_tampering(self, current_frame, window_frames):
                self.tamper_until_frame = max(self.tamper_until_frame, current_frame + window_frames)
        
        track = MockTrack()
        
        # 1. Establish stable linear motion (5 px/frame right)
        curr_x, curr_y = 100, 100
        for i in range(15):
            curr_x += 5
            traj.update(curr_x, curr_y, 5, 0)
            score = agent.evaluate(traj, 1.0, track=track, current_frame=i)
            mcv = agent.mcv_engine.compute(traj, 1.0)
            f.write(f"Frame {i:02d} | Pos: ({curr_x:3d}, {curr_y:3d}) | MCV: {mcv:6.3f} | Score: {score:5.3f}\n")

        # 2. Injection: Sudden 200px teleportation
        f.write("-" * 60 + "\n")
        f.write(" !!! INJECTING 200px TELEPORTATION !!!\n")
        f.write("-" * 60 + "\n")
        
        curr_x += 200
        traj.update(curr_x, curr_y, 200, 0) # Massive velocity spike
        
        mcv = agent.mcv_engine.compute(traj, 1.0)
        score = agent.evaluate(traj, 1.0, track=track, current_frame=15)
        
        f.write(f"Frame 15 | Pos: ({curr_x:3d}, {curr_y:3d}) | MCV: {mcv:6.3f} | Score: {score:5.3f}  <-- ANOMALY\n")
        
        # 3. Post-anomaly recovery (Verify LATCH/BLACKOUT)
        for i in range(16, 26):
            curr_x += 5
            traj.update(curr_x, curr_y, 5, 0)
            mcv = agent.mcv_engine.compute(traj, 1.0)
            score = agent.evaluate(traj, 1.0, track=track, current_frame=i)
            f.write(f"Frame {i:02d} | Pos: ({curr_x:3d}, {curr_y:3d}) | MCV: {mcv:6.3f} | Score: {score:5.3f}\n")

    print(f"Spike test complete. Results written to {output_path}")

if __name__ == "__main__":
    run_test()
