# coherex/detection/yolo_detector.py

from ultralytics import YOLO
import numpy as np
from coherex.config import CONFIG


class YOLODetector:
    """
    Lightweight YOLO-based object detector.
    Responsible ONLY for per-frame object detection.
    """

    def __init__(self, model_path="yolov8n.pt", conf_threshold=None, config=None):
        self.model = YOLO(model_path)
        cfg = config or CONFIG
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None
            else cfg.detection.confidence_threshold
        )

    def detect(self, frame):
        """
        Perform object detection on a single frame.

        Args:
            frame (np.ndarray): BGR image

        Returns:
            List[dict]: detections with bbox, confidence, class_id
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id": class_id
                })

        return detections
