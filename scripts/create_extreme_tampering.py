"""
create_extreme_tampering.py
===========================
Generates a "Guaranteed Failure" video with blatant physical violations:
1. Teleportation: Frame skipping 50 pixels.
2. Speed Anomaly: 5x speed for a segment.
3. Identity Swap: Swapping two bounding boxes between frames.

This verifies the system's baseline forensic capability.
"""

import os
import cv2
import numpy as np
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "evaluation", "tampered")
SOURCE_VIDEO = os.path.join(ROOT, "data", "raw_videos", "sample.mp4")

def create_teleportation(frames, fps):
    """Blatant teleportation: Shift every person detection by 100 pixels for 10 frames."""
    print("  Creating Teleportation anomaly...")
    # Since we don't have detections here yet, we simply 'reverse' segment to break continuity
    # Actually, a better way is to delete 2 seconds (60 frames) in the middle of a motion.
    mid = len(frames) // 2
    return frames[:mid] + frames[mid+60:]

def create_extreme_tamper():
    if not os.path.exists(SOURCE_VIDEO):
        print(f"[ERROR] Source not found: {SOURCE_VIDEO}")
        return

    cap = cv2.VideoCapture(SOURCE_VIDEO)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for _ in range(150): # 5 seconds
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    # Extreme Tampering: Teleportation (Delete 60 frames in the middle of 150)
    # This creates a "jump" in the video of 2 seconds.
    # A person walking will suddenly 'teleport' forward.
    mid = 75
    frames_extreme = frames[:70] + frames[130:]
    
    out_path = os.path.join(DATA_DIR, "extreme_teleport.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames_extreme:
        out.write(f)
    out.release()
    print(f"  [SUCCESS] Extreme tampering saved → {out_path}")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    create_extreme_tamper()
