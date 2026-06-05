import cv2
from collections import deque
import os


class ClipExtractor:
    """
    Extracts evidence clips around tampering events.
    """

    def __init__(self, fps, pre_sec=2, post_sec=2):
        self.fps = fps
        self.pre_frames = int(pre_sec * fps)
        self.post_frames = int(post_sec * fps)

        self.buffer = deque(maxlen=self.pre_frames)
        self.recording = False
        self.frames_left = 0
        self.writer = None
        self.clip_count = 0

    def update(self, frame, state, bbox, output_dir):
        """
        state: COHERENT / SUSPICIOUS / INCONSISTENT
        bbox: (x1, y1, x2, y2)
        """

        self.buffer.append(frame.copy())

        # 🔴 Trigger on SUSPICIOUS or INCONSISTENT
        if state in ("SUSPICIOUS", "INCONSISTENT") and not self.recording:
            self._start_clip(frame.shape[:2], output_dir, bbox)

        if self.recording:
            cropped = self._crop(frame, bbox)
            self.writer.write(cropped)
            self.frames_left -= 1

            if self.frames_left <= 0:
                self._end_clip()

    def _start_clip(self, frame_hw, output_dir, bbox):
        h, w = frame_hw
        x1, y1, x2, y2 = map(int, bbox)

        # Safety clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        clip_w = x2 - x1
        clip_h = y2 - y1

        if clip_w <= 0 or clip_h <= 0:
            return

        os.makedirs(output_dir, exist_ok=True)

        self.clip_count += 1
        clip_path = os.path.join(
            output_dir, f"tampered_clip_{self.clip_count}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            clip_path, fourcc, self.fps, (clip_w, clip_h)
        )

        # Write pre-event frames
        for f in self.buffer:
            self.writer.write(self._crop(f, bbox))

        self.recording = True
        self.frames_left = self.post_frames

    def _crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]

    def _end_clip(self):
        self.writer.release()
        self.writer = None
        self.recording = False
