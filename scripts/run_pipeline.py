import cv2
import os
import sys
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.config import CONFIG as DEFAULT_CONFIG
from coherex.config_loader import load_config_from_yaml
from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore
from coherex.coherence.mcv import MotionCoherenceEngine
from coherex.coherence.interpreter import MotionInterpreter
from coherex.output.visualizer import Visualizer
from coherex.output.clipper import ClipExtractor
from ultralytics import YOLO
from coherex.tracking.pose import compute_pose_scale

# --- Integrity agent framework ---
from coherex.integrity.motion_agent import MotionAgent
from coherex.integrity.continuity_agent import ContinuityAgent
from coherex.integrity.crowd_agent import CrowdAgent
from coherex.integrity.fusion_engine import IntegrityFusionEngine
from coherex.integrity.segment_aggregator import SegmentIntegrityAggregator
from coherex.integrity.reliability import ReliabilityEstimator


# ===================== PATHS =====================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CLIPS_DIR = os.path.join(ROOT_DIR, "data", "clips", "raw")

# =================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoheRex Integrity Pipeline"
    )
    parser.add_argument(
        "--video", type=str,
        default=os.path.join(ROOT_DIR, "data", "raw_videos", "sample1.mp4"),
        help="Path to input video file",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config override file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for annotated output video",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    if args.config:
        CONFIG = load_config_from_yaml(args.config)
        print(f"[CONFIG] Loaded overrides from: {args.config}")
    else:
        CONFIG = DEFAULT_CONFIG

    VIDEO_PATH = args.video
    OUTPUT_VIDEO_PATH = args.output or os.path.join(
        ROOT_DIR, "data", "processed_videos", "annotated_output3.mp4"
    )

    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    if os.path.exists(RAW_CLIPS_DIR) and not os.path.isdir(RAW_CLIPS_DIR):
        os.remove(RAW_CLIPS_DIR)
    os.makedirs(RAW_CLIPS_DIR, exist_ok=True)

    print(f"[CONFIG] {CONFIG.version}")
    print(f"[CONFIG] Detection threshold: {CONFIG.detection.confidence_threshold}")
    print(f"[CONFIG] Tracking distance:   {CONFIG.tracking.max_association_distance}")
    print(f"[CONFIG] Fusion weights:       {CONFIG.integrity.fusion_weights}")
    print(f"[CONFIG] Segment window:       {CONFIG.integrity.segment_window_frames}")
    print(f"[CONFIG] Verdict thresholds:   {CONFIG.integrity.verdict_thresholds}")

    # ------------------ INITIALIZE ------------------

    detector = YOLODetector(config=CONFIG)
    pose_model = YOLO("yolov8n-pose.pt")
    tracker = TrackManager(config=CONFIG)
    trajectory_store = TrajectoryStore(config=CONFIG)
    mcv_engine = MotionCoherenceEngine()
    interpreter = MotionInterpreter(config=CONFIG)
    visualizer = Visualizer()

    # --- Integrity agents (all receive config) ---
    motion_agent = MotionAgent(config=CONFIG)
    continuity_agent = ContinuityAgent(config=CONFIG)
    crowd_agent = CrowdAgent(config=CONFIG)
    reliability_estimator = ReliabilityEstimator()
    fusion_engine = IntegrityFusionEngine(config=CONFIG, reliability_estimator=reliability_estimator)
    aggregator = SegmentIntegrityAggregator(config=CONFIG)

    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, fps, frame_size
    )

    clipper = ClipExtractor(
        fps=fps,
        pre_sec=5,
        post_sec=5
    )

    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # For reporting
    frame_scores = []
    reliabilities = []

    # ------------------ MAIN LOOP ------------------

    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. DETECTION
        detections = detector.detect(frame)

        person_centers = []
        person_boxes = []

        for det in detections:
            if det["class_id"] == CONFIG.detection.person_class_id:
                x1, y1, x2, y2 = det["bbox"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                person_centers.append((cx, cy))
                person_boxes.append(det["bbox"])

        # 2. TRACKING
        
        # Extract pose
        pose_results = pose_model(frame, verbose=False)
        pose_keypoints = {}
        
        for r in pose_results:
            if r.keypoints is None:
                continue
            for i, kp in enumerate(r.keypoints.xy):
                pose_keypoints[i] = kp.cpu().numpy()
        
        tracker.current_frame = frame
        tracker.current_pose = pose_keypoints
        
        tracks = tracker.update(
            detections=person_centers,
            bboxes=person_boxes,
            frame_id=frame_id
        )

        # 3. TRAJECTORY UPDATE
        trajectory_store.update_from_tracks(tracks)

        # 4. COHERENCE + INTERPRETATION + INTEGRITY AGENTS
        motion_scores = []
        continuity_scores = []

        for track in tracks:
            state = trajectory_store.get_state(track.track_id)
            if state is None:
                continue

            # --- Legacy MCV path ---
            mcv = mcv_engine.compute(state, track.confidence)
            motion_state = interpreter.interpret(mcv)

            # HARD override for semantic tampering
            if frame_id <= track.tamper_until_frame:
                motion_state = "INCONSISTENT"

            # --- Integrity agent scores ---
            m_score = motion_agent.evaluate(state, track.confidence)
            c_score = continuity_agent.evaluate(track, frame_id)
            motion_scores.append(m_score)
            continuity_scores.append(c_score)

            # 5. FIND NEAREST BBOX
            bbox = None
            min_dist = float("inf")

            for box in person_boxes:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dist = (track.x - cx) ** 2 + (track.y - cy) ** 2
                if dist < min_dist:
                    min_dist = dist
                    bbox = box

            if bbox is not None:
                visualizer.draw(frame, track, bbox, motion_state)

        # 6. CROWD + FUSION + SEGMENT AGGREGATION
        crowd_score = crowd_agent.evaluate(tracks)
        integrity_score = fusion_engine.fuse(
            motion_scores,
            continuity_scores,
            crowd_score,
            tracks=tracks,
            trajectory_store=trajectory_store
        )
        r_motion, r_cont, r_crowd = fusion_engine.get_last_reliabilities()
        reliabilities.append((r_motion, r_cont, r_crowd))
        frame_scores.append(integrity_score)
        aggregator.update(integrity_score)

        segment_score = aggregator.get_segment_score()
        label = fusion_engine.interpret(integrity_score)

        # Colour coding: GREEN = HIGH, YELLOW = MODERATE, RED = COMPROMISED
        colour_map = {
            "HIGH":        (0, 220, 80),
            "MODERATE":    (0, 200, 255),
            "COMPROMISED": (0, 60, 255),
        }
        overlay_colour = colour_map.get(label, (255, 255, 255))

        # Frame-level integrity overlay (top-left)
        cv2.putText(
            frame,
            f"Integrity: {integrity_score:.2f}  [{label}]",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            overlay_colour,
            2,
        )

        # Segment-level score overlay (below frame integrity)
        cv2.putText(
            frame,
            f"Segment:   {segment_score:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 220, 0),
            2,
        )

        # Reliability overlay (top-right)
        rel_text = f"Reliability: M={r_motion:.2f} C={r_cont:.2f} Cr={r_crowd:.2f}"
        cv2.putText(
            frame,
            rel_text,
            (width - 450, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            2,
        )

        # 7. WRITE FRAME
        writer.write(frame)
        frame_id += 1

    # ------------------ CLEANUP ------------------

    cap.release()
    writer.release()

    print("Processing complete.")
    print("Annotated video saved to:", OUTPUT_VIDEO_PATH)
    print("Raw tampered clips saved in:", RAW_CLIPS_DIR)


if __name__ == "__main__":
    main()
