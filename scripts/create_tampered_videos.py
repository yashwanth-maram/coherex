"""
create_tampered_videos.py
=========================
Builds the CoheRex evaluation dataset from source videos.

Outputs:
    data/evaluation/authentic/  — 20 authentic clips (unmodified segments)
    data/evaluation/tampered/   — 20 tampered clips (5 tampering types × 4 seeds)

Tampering types implemented:
    1. frame_deletion     — remove N consecutive frames
    2. frame_duplication  — repeat a short segment
    3. speed_manipulation — accelerate a window (drop every 2nd frame, then pad)
    4. clip_splice        — insert frames from the other source video
    5. reverse_segment    — reverse a short segment (temporal order violation)

Usage:
    python scripts/create_tampered_videos.py
"""

import cv2
import os
import sys
import random
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SRC_A = os.path.join(ROOT, "data", "raw_videos", "sample.mp4")
SRC_B = os.path.join(ROOT, "data", "raw_videos", "sample1.mp4")
AUTH_DIR  = os.path.join(ROOT, "data", "evaluation", "authentic")
TAMP_DIR  = os.path.join(ROOT, "data", "evaluation", "tampered")

CLIPS_PER_SRC = 15         # try to get 15 from SRC_A
CLIP_DURATION_SEC = 3      # 3-second clips — maximizes coverage from short videos
TAMPERS_PER_TYPE = 4       # 5 types × 4 variants = 20 tampered

random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_all_frames(path: str):
    """Read every frame from a video into a list."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_frames(frames, out_path: str, fps: float, frame_size=None):
    """Write a list of frames to an mp4 file."""
    if not frames:
        print(f"  [WARN] no frames — skipping {out_path}")
        return
    h, w = frames[0].shape[:2]
    size = frame_size or (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, size)
    for f in frames:
        if f.shape[1] != size[0] or f.shape[0] != size[1]:
            f = cv2.resize(f, size)
        writer.write(f)
    writer.release()


def extract_segment(frames, fps, start_sec, duration_sec):
    """Extract a clean segment from a frame list."""
    start = int(start_sec * fps)
    end   = min(int((start_sec + duration_sec) * fps), len(frames))
    seg   = frames[start:end]
    return seg if len(seg) > 30 else None


# ──────────────────────────────────────────────────────────────────────────────
# Tampering functions  (operate on a frame list, return modified list)
# ──────────────────────────────────────────────────────────────────────────────

def tamper_frame_deletion(frames, fps, seed=0):
    """Remove 5–20 consecutive frames starting at a random interior point."""
    rng = random.Random(seed)
    n_del = rng.randint(5, min(20, len(frames) // 4))
    start = rng.randint(10, max(11, len(frames) - n_del - 10))
    result = frames[:start] + frames[start + n_del:]
    print(f"    frame_deletion: removed frames [{start}:{start+n_del}] ({n_del} frames)")
    return result


def tamper_frame_duplication(frames, fps, seed=0):
    """Duplicate a 1-second segment at a random interior point."""
    rng = random.Random(seed)
    dup_len = int(fps * 1.5)
    start = rng.randint(10, max(11, len(frames) - dup_len - 10))
    dup   = frames[start:start + dup_len]
    result = frames[:start] + dup + dup + frames[start + dup_len:]
    print(f"    frame_duplication: duplicated [{start}:{start+dup_len}] ({dup_len} frames)")
    return result


def tamper_speed_manipulation(frames, fps, seed=0):
    """Simulate 1.5× speed in a 2-second window by dropping every 2nd frame."""
    rng = random.Random(seed)
    win_len = int(fps * 2.0)
    start = rng.randint(10, max(11, len(frames) - win_len - 10))
    fast  = frames[start:start + win_len:2]           # keep every 2nd → 1.5× speed
    pad   = [frames[start + win_len - 1]] * len(fast) # pad to maintain length
    result = frames[:start] + fast + pad + frames[start + win_len:]
    print(f"    speed_manipulation: accelerated [{start}:{start+win_len}] (dropped {len(fast)} frames)")
    return result


def tamper_clip_splice(frames_a, frames_b, fps, seed=0):
    """Insert 2–3 seconds from frames_b into frames_a."""
    rng = random.Random(seed)
    splice_len = int(fps * rng.uniform(2.0, 3.0))
    # Insert point
    insert_at = rng.randint(10, max(11, len(frames_a) - 10))
    # Source point in B
    b_start = rng.randint(0, max(0, len(frames_b) - splice_len - 1))
    foreign = frames_b[b_start:b_start + splice_len]
    # Resize foreign frames to match A's resolution
    h, w = frames_a[0].shape[:2]
    foreign = [cv2.resize(f, (w, h)) for f in foreign]
    result = frames_a[:insert_at] + foreign + frames_a[insert_at:]
    print(f"    clip_splice: inserted {splice_len} frames from SRC_B at position {insert_at}")
    return result


def tamper_reverse_segment(frames, fps, seed=0):
    """Reverse a 1.5-second segment — creates impossible temporal flow."""
    rng = random.Random(seed)
    rev_len = int(fps * 1.5)
    start = rng.randint(10, max(11, len(frames) - rev_len - 10))
    rev_seg = frames[start:start + rev_len][::-1]
    result = frames[:start] + rev_seg + frames[start + rev_len:]
    print(f"    reverse_segment: reversed [{start}:{start+rev_len}] ({rev_len} frames)")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main generation
# ──────────────────────────────────────────────────────────────────────────────

TAMPER_FNS = {
    "frame_deletion":     tamper_frame_deletion,
    "frame_duplication":  tamper_frame_duplication,
    "speed_manipulation": tamper_speed_manipulation,
    "clip_splice":        None,          # special — needs frames_b
    "reverse_segment":    tamper_reverse_segment,
}


def main():
    os.makedirs(AUTH_DIR, exist_ok=True)
    os.makedirs(TAMP_DIR, exist_ok=True)

    # ── 1. Load source videos ────────────────────────────────────────────────
    print("[LOAD] Reading source videos …")
    frames_a, fps_a = read_all_frames(SRC_A)
    frames_b, fps_b = read_all_frames(SRC_B)
    print(f"  SRC_A: {len(frames_a)} frames @ {fps_a:.1f} fps")
    print(f"  SRC_B: {len(frames_b)} frames @ {fps_b:.1f} fps")

    fps = fps_a   # use A's fps throughout

    total_dur_a = len(frames_a) / fps
    total_dur_b = len(frames_b) / fps

    # ── 2. Generate authentic clips ──────────────────────────────────────────
    print("\n[AUTH] Generating authentic clips …")
    auth_count = 0
    # From SRC_A: CLIPS_PER_SRC clips
    step_a = max(CLIP_DURATION_SEC + 1, (total_dur_a - CLIP_DURATION_SEC) / CLIPS_PER_SRC)
    for i in range(CLIPS_PER_SRC):
        start = i * step_a
        seg = extract_segment(frames_a, fps, start, CLIP_DURATION_SEC)
        if seg is None:
            print(f"  [WARN] SRC_A segment {i} too short, skipping")
            continue
        auth_count += 1
        out_path = os.path.join(AUTH_DIR, f"auth_{auth_count:02d}.mp4")
        write_frames(seg, out_path, fps)
        print(f"  auth_{auth_count:02d}.mp4 — {len(seg)} frames from SRC_A @ t={start:.1f}s")

    # From SRC_B: use all available non-overlapping segments (may be <10)
    # SRC_B may be very short — use 1-second step to maximize clips
    b_step = max(1.0, CLIP_DURATION_SEC)
    b_start = 0.0
    while auth_count < 20 and b_start + CLIP_DURATION_SEC <= total_dur_b:
        seg = extract_segment(frames_b, fps_b, b_start, CLIP_DURATION_SEC)
        if seg is None:
            break
        auth_count += 1
        out_path = os.path.join(AUTH_DIR, f"auth_{auth_count:02d}.mp4")
        write_frames(seg, out_path, fps_b)
        print(f"  auth_{auth_count:02d}.mp4 — {len(seg)} frames from SRC_B @ t={b_start:.1f}s")
        b_start += b_step

    # If still under 20, generate more from SRC_A (overlapping starts)
    if auth_count < 20:
        print(f"  [INFO] SRC_B short. Generating {20 - auth_count} extra clips from SRC_A …")
        extra_step = max(CLIP_DURATION_SEC, total_dur_a / 30)
        extra_idx = 0
        while auth_count < 20:
            start = (CLIPS_PER_SRC * step_a) + extra_idx * extra_step
            if start + CLIP_DURATION_SEC > total_dur_a:
                break
            seg = extract_segment(frames_a, fps, start, CLIP_DURATION_SEC)
            if seg is None:
                break
            auth_count += 1
            out_path = os.path.join(AUTH_DIR, f"auth_{auth_count:02d}.mp4")
            write_frames(seg, out_path, fps)
            print(f"  auth_{auth_count:02d}.mp4 — {len(seg)} frames from SRC_A (extra) @ t={start:.1f}s")
            extra_idx += 1

    print(f"  → {auth_count} authentic clips generated")

    # ── 3. Generate tampered clips ───────────────────────────────────────────
    print("\n[TAMP] Generating tampered clips …")
    tamp_count = 0

    # Use midpoints of SRC_A as base segments for tampering
    base_starts = []
    for i in range(TAMPERS_PER_TYPE):
        base_starts.append(i * (total_dur_a / TAMPERS_PER_TYPE))

    for ttype, fn in TAMPER_FNS.items():
        for variant_idx in range(TAMPERS_PER_TYPE):
            seed = variant_idx * 100 + list(TAMPER_FNS.keys()).index(ttype)
            start = base_starts[variant_idx]
            base_seg = extract_segment(frames_a, fps, start, CLIP_DURATION_SEC)
            if base_seg is None:
                print(f"  [WARN] base segment for {ttype} variant {variant_idx} too short")
                continue

            print(f"  Generating {ttype} variant {variant_idx+1} (seed={seed}) …")
            try:
                if ttype == "clip_splice":
                    modified = tamper_clip_splice(base_seg, frames_b, fps, seed=seed)
                else:
                    modified = fn(base_seg, fps, seed=seed)
            except Exception as e:
                print(f"    [ERROR] {e}")
                continue

            tamp_count += 1
            out_path = os.path.join(TAMP_DIR, f"tamp_{tamp_count:02d}.mp4")
            write_frames(modified, out_path, fps)
            print(f"    → tamp_{tamp_count:02d}.mp4 ({len(modified)} frames)")

    print(f"\n  → {tamp_count} tampered clips generated")

    # ── 4. Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"  Authentic : {auth_count:>3}  → {AUTH_DIR}")
    print(f"  Tampered  : {tamp_count:>3}  → {TAMP_DIR}")
    print(f"  Total     : {auth_count + tamp_count:>3}")
    print()
    print("Tampering types:")
    for t in TAMPER_FNS:
        print(f"  {t:<25} × {TAMPERS_PER_TYPE} variants")
    print()
    print("Dataset ready. Run scripts/evaluate_dataset.py next.")


if __name__ == "__main__":
    main()
