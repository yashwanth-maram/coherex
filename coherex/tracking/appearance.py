
import cv2
import numpy as np


def extract_color_histogram(frame, bbox, bins=(16, 16, 16)):
    """
    Extract normalized HSV color histogram from upper-body region.
    """
    x1, y1, x2, y2 = map(int, bbox)

    if x2 <= x1 or y2 <= y1:
        return None

    h = y2 - y1
    upper_body = frame[y1 : y1 + int(0.6 * h), x1:x2]

    if upper_body.size == 0:
        return None

    hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None, bins,
        [0, 180, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)
    return hist


def histogram_similarity(h1, h2):
    """
    Compare histograms using cosine similarity.
    """
    if h1 is None or h2 is None:
        return 0.0

    h1 = h1.flatten()
    h2 = h2.flatten()

    denom = (np.linalg.norm(h1) * np.linalg.norm(h2))
    if denom == 0:
        return 0.0

    return float(np.dot(h1, h2) / denom)
