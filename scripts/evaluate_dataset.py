"""
evaluate_dataset.py
===================
Batch evaluation script for the CoheRex scientific validation layer.

For each video in data/evaluation/authentic/ and data/evaluation/tampered/,
runs the full CoheRex integrity pipeline (no video output written),
extracts the mean segment integrity score, and writes results.csv.

Usage:
    python scripts/evaluate_dataset.py [--window 150] [--agents full]

    --window  : Segment window size in frames (default: 150)
    --agents  : Which agents to use — one of:
                  motion | motion_continuity | motion_crowd | full
                (default: full)
    --config  : Optional path to a YAML config override file
"""

import cv2
import os
import sys
import csv
import argparse
from tqdm import tqdm
from scipy.stats import kurtosis
from skimage.metrics import structural_similarity as ssim
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from coherex.config import CONFIG as DEFAULT_CONFIG, IntegrityConfig, SystemConfig
from coherex.config_loader import load_config_from_yaml
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore
from coherex.coherence.mcv import MotionCoherenceEngine
from coherex.coherence.interpreter import MotionInterpreter
from coherex.integrity.motion_agent import MotionAgent
from coherex.integrity.continuity_agent import ContinuityAgent
from coherex.integrity.crowd_agent import CrowdAgent
from coherex.integrity.fusion_engine import IntegrityFusionEngine
from coherex.integrity.segment_aggregator import SegmentIntegrityAggregator
from coherex.integrity.reliability import ReliabilityEstimator
from coherex.tracking.pose import compute_pose_scale
from ultralytics import YOLO


# ─── directories ──────────────────────────────────────────────────────────────
AUTH_DIR    = os.path.join(ROOT, "data", "evaluation", "authentic")
TAMP_DIR    = os.path.join(ROOT, "data", "evaluation", "tampered")
RESULTS_CSV = os.path.join(ROOT, "data", "evaluation", "results.csv")


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CoheRex Batch Evaluator")
    p.add_argument("--window", type=int, default=150,
                   help="Segment aggregation window (frames)")
    p.add_argument("--agents", type=str, default="full",
                   choices=["motion", "motion_continuity", "motion_crowd", "full"],
                   help="Agent configuration for ablation study")
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config override")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: data/evaluation/results.csv)")
    p.add_argument("--input", type=str, default=None,
                   help="Input directory overrides (must contain authentic/ and tampered/ subdirs)")
    p.add_argument("--append", action="store_true", default=False,
                   help="Append to existing CSV instead of overwriting (default: overwrite)")
    return p.parse_args()


# ─── Single-video evaluation ───────────────────────────────────────────────────

def evaluate_video(
    video_path: str,
    config,
    window_size: int,
    agents_mode: str,
    pose_model,
) -> float:
    """
    Run the CoheRex pipeline on a single video.
    Returns the mean segment integrity score.
    """

    detector = YOLODetector(config=config)
    tracker  = TrackManager(config=config)
    traj_store = TrajectoryStore(config=config)
    mcv_engine = MotionCoherenceEngine()
    interpreter = MotionInterpreter(config=config)

    motion_agent     = MotionAgent(config=config)
    continuity_agent = ContinuityAgent(config=config)
    crowd_agent      = CrowdAgent(config=config)
    rel_estimator    = ReliabilityEstimator()
    fusion_engine    = IntegrityFusionEngine(config=config, reliability_estimator=rel_estimator)
    aggregator       = SegmentIntegrityAggregator(window_size=window_size, config=config)

    use_continuity = agents_mode in ("motion_continuity", "full")
    use_crowd      = agents_mode in ("motion_crowd", "full")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    all_segment_scores = []
    
    # Debug tracking
    frame_integrity_scores = []
    all_motion_scores = []
    all_continuity_scores = []
    all_crowd_scores = []
    all_crowd_scores = []
    all_raw_mcvs = []
    
    # Hybrid Features
    prev_gray = None
    all_ssim_scores = []
    all_flow_variances = []
    all_track_accelerations = []
    
    # Burst detection
    current_burst = 0
    max_burst = 0
    frames_with_anomaly = 0

    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        detections = detector.detect(frame)
        person_centers, person_boxes = [], []
        for det in detections:
            if det["class_id"] == config.detection.person_class_id:
                x1, y1, x2, y2 = det["bbox"]
                person_centers.append(((x1+x2)/2, (y1+y2)/2))
                person_boxes.append(det["bbox"])

        # Pose
        pose_results = pose_model(frame, verbose=False)
        pose_keypoints = {}
        for r in pose_results:
            if r.keypoints is None:
                continue
            for i, kp in enumerate(r.keypoints.xy):
                pose_keypoints[i] = kp.cpu().numpy()

        tracker.current_frame = frame
        tracker.current_pose  = pose_keypoints
        tracks = tracker.update(
            detections=person_centers,
            bboxes=person_boxes,
            frame_id=frame_id
        )

        traj_store.update_from_tracks(tracks)

        # A) Optical Flow Variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            # Dense Optical Flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            all_flow_variances.append(np.var(flow_mag))
            
            # B) SSIM Temporal Drift
            score, _ = ssim(prev_gray, gray, full=True)
            all_ssim_scores.append(score)
        
        prev_gray = gray

        motion_scores, continuity_scores = [], []
        for track in tracks:
            state = traj_store.get_state(track.track_id)
            if state is None:
                continue
            m_score = motion_agent.evaluate(state, track.confidence, track=track, current_frame=frame_id)
            
            # Collect raw MCV for per-video max reporting
            raw_mcv = motion_agent.mcv_engine.compute(state, track.confidence)
            all_raw_mcvs.append(abs(raw_mcv))
            # Penalty for 'Late Birth': appearing after the video start is suspicious in surveillance
            # but legitimate at frame 0.
            if track.first_seen_frame > 5 and state.total_age < 30:
                m_score *= 0.5
            
            motion_scores.append(m_score)
            
            if use_continuity:
                c_score = continuity_agent.evaluate(track, frame_id)
                continuity_scores.append(c_score)
            else:
                continuity_scores.append(1.0)
            
            # Collect accelerations for Kurtosis
            if len(state.accelerations) > 0:
                all_track_accelerations.append(abs(state.accelerations[-1]))
        
        # Physics-based anomaly detection (Step 1 & 3)
        # Any track with MCV > 2.0 triggers a frame-level anomaly flag
        frame_has_physics_anomaly = False
        for track in tracks:
            state = traj_store.get_state(track.track_id)
            if state:
                raw_mcv = mcv_engine.compute(state, track.confidence)
                if abs(raw_mcv) > 2.0:
                    frame_has_physics_anomaly = True
                    break
        
        if frame_has_physics_anomaly:
            frames_with_anomaly += 1
            current_burst += 1
            max_burst = max(max_burst, current_burst)
        else:
            current_burst = 0

        crowd_score = crowd_agent.evaluate(tracks) if use_crowd else 1.0

        integrity_score = fusion_engine.fuse(
            motion_scores,
            continuity_scores,
            crowd_score,
            tracks=tracks,
            trajectory_store=traj_store,
        )

        aggregator.update(integrity_score)
        
        # Track for debug
        frame_integrity_scores.append(integrity_score)
        all_segment_scores.append(aggregator.get_segment_score())
        
        if motion_scores:
            all_motion_scores.append(sum(motion_scores)/len(motion_scores))
        if continuity_scores:
            all_continuity_scores.append(sum(continuity_scores)/len(continuity_scores))
        all_crowd_scores.append(crowd_score)
        
        if "extreme_teleport" in video_path and frame_id < 10:
            if motion_scores:
                m_avg = sum(motion_scores)/len(motion_scores)
                tqdm.write(f"  [DEBUG] Frame {frame_id:03d} Motion Score: {m_avg:.4f}")

        frame_id += 1

    cap.release()

    if not all_segment_scores:
        return {
            "mean_segment": 1.0, "min_frame": 1.0, 
            "motion": 1.0, "continuity": 1.0, "crowd": 1.0,
            "perc_below": 0.0
        }

    return {
        "mean_segment":  sum(all_segment_scores) / len(all_segment_scores),
        "min_frame":     min(frame_integrity_scores),
        "motion":        sum(all_motion_scores) / len(all_motion_scores) if all_motion_scores else 1.0,
        "motion_min":    min(all_motion_scores) if all_motion_scores else 1.0,
        "continuity":    sum(all_continuity_scores) / len(all_continuity_scores) if all_continuity_scores else 1.0,
        "crowd":         sum(all_crowd_scores) / len(all_crowd_scores) if all_crowd_scores else 1.0,
        "perc_below":    sum(1 for s in frame_integrity_scores if s < 0.6) / len(frame_integrity_scores),
        "anomaly_density": frames_with_anomaly / frame_id if frame_id > 0 else 0.0,
        "max_mcv":       max(all_raw_mcvs) if all_raw_mcvs else 0.0,
        "burst_length":  max_burst,
        "ssim_mean":     sum(all_ssim_scores) / len(all_ssim_scores) if all_ssim_scores else 1.0,
        "flow_var":      sum(all_flow_variances) / len(all_flow_variances) if all_flow_variances else 0.0,
        "accel_kurt":    float(kurtosis(all_track_accelerations)) if len(all_track_accelerations) > 4 else 0.0,
        "volatility":    np.std(frame_integrity_scores) if frame_integrity_scores else 0.0,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Config
    if args.config:
        config = load_config_from_yaml(args.config)
        print(f"[CONFIG] Loaded from {args.config}")
    else:
        config = DEFAULT_CONFIG

    output_csv = args.output or RESULTS_CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # ── Input Overrides ──────────────────────────────────────────────────
    auth_dir = AUTH_DIR
    tamp_dir = TAMP_DIR
    if args.input:
        auth_dir = os.path.join(args.input, "authentic")
        tamp_dir = os.path.join(args.input, "tampered")

    print(f"[EVAL] Window size : {args.window}")
    print(f"[EVAL] Agents mode : {args.agents}")
    print(f"[EVAL] Output CSV  : {output_csv}")

    # Shared pose model (expensive to build — load once)
    pose_model = YOLO(os.path.join(ROOT, "yolov8n-pose.pt"))

    # Gather video paths  (label: 0=authentic, 1=tampered)
    def collect(directory, label):
        entries = []
        if not os.path.isdir(directory):
            print(f"  [WARN] Directory not found: {directory}")
            return entries
        for fname in sorted(os.listdir(directory)):
            if fname.lower().endswith(".mp4"):
                entries.append((fname, os.path.join(directory, fname), label))
        return entries

    videos = collect(auth_dir, 0) + collect(tamp_dir, 1)
    if not videos:
        print("[ERROR] No videos found. Run scripts/create_tampered_videos.py first.")
        sys.exit(1)

    print(f"\n[EVAL] Processing {len(videos)} videos …\n")

    fieldnames = ["video_name", "integrity_score", "min_integrity", "volatility",
                 "motion_mean", "motion_min", "continuity_mean", "crowd_mean", 
                 "perc_below_06", "anomaly_density", "max_mcv", "burst_length", 
                 "ssim_mean", "flow_var", "accel_kurt", "label"]

    write_mode = "a" if args.append else "w"  # default: overwrite
    with open(output_csv, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # always write header (fresh CSV)

        for fname, vpath, label in tqdm(videos, desc="Evaluating"):
            try:
                res_dict = evaluate_video(
                    video_path=vpath,
                    config=config,
                    window_size=args.window,
                    agents_mode=args.agents,
                    pose_model=pose_model,
                )
                score = res_dict["mean_segment"]
            except Exception as e:
                print(f"\n  [ERROR] {fname}: {e}")
                res_dict = None
                score = -1.0

            if res_dict:
                row = {
                    "video_name":      fname,
                    "integrity_score": round(score, 6),
                    "min_integrity":   round(res_dict["min_frame"], 6),
                    "volatility":      round(res_dict["volatility"], 6),
                    "motion_mean":     round(res_dict["motion"], 6),
                    "motion_min":      round(res_dict["motion_min"], 6),
                    "continuity_mean": round(res_dict["continuity"], 6),
                    "crowd_mean":      round(res_dict["crowd"], 6),
                    "perc_below_06":   round(res_dict["perc_below"], 6),
                    "anomaly_density": round(res_dict["anomaly_density"], 6),
                    "max_mcv":         round(res_dict["max_mcv"], 6),
                    "burst_length":    res_dict["burst_length"],
                    "ssim_mean":       round(res_dict["ssim_mean"], 6),
                    "flow_var":        round(res_dict["flow_var"], 6),
                    "accel_kurt":      round(res_dict["accel_kurt"], 6),
                    "label":           label,
                }
                tqdm.write(f"  {fname:25s}  score={score:.4f}  mcv={res_dict['max_mcv']:.2f}  kurt={res_dict['accel_kurt']:.2f}  label={label}")
            else:
                row = {
                    "video_name":      fname,
                    "integrity_score": -1.0,
                    "min_integrity":   -1.0,
                    "motion_mean":     -1.0,
                    "continuity_mean": -1.0,
                    "crowd_mean":      -1.0,
                    "perc_below_06":   -1.0,
                    "anomaly_density": -1.0,
                    "max_mcv":         -1.0,
                    "burst_length":    -1,
                    "ssim_mean":       -1.0,
                    "flow_var":        -1.0,
                    "accel_kurt":      -1.0,
                    "label":           label,
                }
            
            writer.writerow(row)
            f.flush() # Ensure it's written

    print(f"\n[DONE] Results saved to {output_csv}")
    print(f"       Next step: python scripts/train_meta_classifier.py")


if __name__ == "__main__":
    main()
