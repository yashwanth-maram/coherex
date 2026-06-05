import cv2
import os


class ForensicUpscaler:
    """
    Forensic-safe video upscaling for visualization.
    Does NOT modify original evidence.
    """

    def __init__(self, scale=2):
        self.scale = scale

    def upscale_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_w, out_h = w * self.scale, h * self.scale

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (out_w, out_h)
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            up = cv2.resize(
                frame,
                (out_w, out_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            # CLAHE for contrast enhancement (forensic-safe)
            lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            writer.write(enhanced)

        cap.release()
        writer.release()
