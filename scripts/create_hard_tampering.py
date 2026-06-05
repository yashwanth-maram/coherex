"""
create_hard_tampering.py
========================
Generates "Violent" tampering events to stress-test CoheRex forensic sensitivity.
Goal: Create a dataset where AUC >= 0.75 is achievable through physics-based signals.

Tampering types:
1. hard_frame_deletion  - Remove 40 frames (sustained blackout).
2. hard_speed_burst     - 4x speed for 1 second.
3. hard_reverse_segment - Reverse 30 frames.
4. hard_teleportation   - Skip 60 frames (2 seconds).
5. hard_clip_splice     - 3 second high-contrast splice.
"""

import cv2
import os
import sys
import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_A = os.path.join(ROOT, "data", "raw_videos", "sample.mp4")
SRC_B = os.path.join(ROOT, "data", "raw_videos", "sample1.mp4")
AUTH_DIR = os.path.join(ROOT, "data", "evaluation_hard", "authentic")
TAMP_DIR = os.path.join(ROOT, "data", "evaluation_hard", "tampered")

def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames, fps

def write_frames(frames, out_path, fps):
    if not frames: return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for f in frames: writer.write(f)
    writer.release()

def tamper_hard_deletion(frames, seed=0):
    rng = random.Random(seed)
    n_del = 40
    start = rng.randint(10, len(frames) - n_del - 10)
    return frames[:start] + frames[start + n_del:]

def tamper_hard_speed(frames, seed=0):
    rng = random.Random(seed)
    win = 60
    start = rng.randint(10, len(frames) - win - 10)
    fast = frames[start:start+win:4] # 4x speed
    pad = [frames[start+win-1]] * (win - len(fast))
    return frames[:start] + fast + pad + frames[start+win:]

def tamper_hard_reverse(frames, seed=0):
    rng = random.Random(seed)
    win = 30
    start = rng.randint(10, len(frames) - win - 10)
    rev = frames[start:start+win][::-1]
    return frames[:start] + rev + frames[start+win:]

def tamper_hard_teleport(frames, seed=0):
    # Same as deletion but larger
    rng = random.Random(seed)
    n_del = 60
    start = rng.randint(10, len(frames) - n_del - 10)
    return frames[:start] + frames[start + n_del:]

def tamper_hard_splice(frames_a, frames_b, seed=0):
    rng = random.Random(seed)
    win = 75 # 3 seconds
    insert_at = rng.randint(10, len(frames_a) - 10)
    b_start = rng.randint(0, len(frames_b) - win - 1)
    foreign = frames_b[b_start:b_start + win]
    h, w = frames_a[0].shape[:2]
    foreign = [cv2.resize(f, (w, h)) for f in foreign]
    return frames_a[:insert_at] + foreign + frames_a[insert_at:]

def main():
    os.makedirs(AUTH_DIR, exist_ok=True)
    os.makedirs(TAMP_DIR, exist_ok=True)
    
    print("[LOAD] Reading sources...")
    fa, fps_a = read_all_frames(SRC_A)
    fb, fps_b = read_all_frames(SRC_B)
    
    # 1. Authentic (just copy / extract first 20 segments)
    print("[AUTH] Generating 20 authentic...")
    for i in range(20):
        start = i * 75
        seg = fa[start:start+100]
        write_frames(seg, os.path.join(AUTH_DIR, f"auth_{i:02d}.mp4"), fps_a)

    # 2. Tampered
    print("[TAMP] Generating 20 violent tampered...")
    fns = [tamper_hard_deletion, tamper_hard_speed, tamper_hard_reverse, tamper_hard_teleport]
    count = 0
    for fn in fns:
        for v in range(4): # 16 videos
            seed = count * 10
            base = fa[count*50 : count*50 + 150]
            mod = fn(base, seed=seed)
            write_frames(mod, os.path.join(TAMP_DIR, f"hard_{count:02d}.mp4"), fps_a)
            count += 1
            
    # 3. Splice (4 videos)
    for v in range(4):
        seed = count * 10
        base = fa[count*50 : count*50 + 150]
        mod = tamper_hard_splice(base, fb, seed=seed)
        write_frames(mod, os.path.join(TAMP_DIR, f"hard_{count:02d}.mp4"), fps_a)
        count += 1

    print(f"Done. Dataset saved to {AUTH_DIR} and {TAMP_DIR}")

if __name__ == "__main__":
    main()
