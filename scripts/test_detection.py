import cv2
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coherex.detection.yolo_detector import YOLODetector

VIDEO_PATH = "data/raw_videos/sample.mp4"  # put ONE test video here

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

detector = YOLODetector()

detections = detector.detect(frame)

print("Detections:")
for det in detections:
    print(det)
