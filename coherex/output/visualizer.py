# coherex/output/visualizer.py
"""
CoheRex Visualizer — Rich Tampering Annotation Overlay

Draws per-track forensic annotations directly onto video frames:
  - Color-coded bounding boxes (COHERENT / SUSPICIOUS / INCONSISTENT)
  - Track ID + motion state label
  - Per-frame integrity score + verdict badge
  - Motion dis
  continuity warning stripe
  - AI classifier verdict (when available)
"""

import cv2
import numpy as np


# ─── Color palette (BGR) ────────────────────────────────────────────
CLR_COHERENT     = (50, 220, 80)     # green
CLR_SUSPICIOUS   = (0, 200, 255)     # amber/yellow
CLR_INCONSISTENT = (40, 40, 255)     # red
CLR_WHITE        = (230, 230, 230)
CLR_BLACK        = (0, 0, 0)
CLR_DARK         = (15, 15, 25)


class Visualizer:
    """
    Draws tracking and tampering forensic annotations on frames.

    Usage:
        viz = Visualizer()
        # per-track call:
        viz.draw(frame, track, bbox, motion_state, motion_score, continuity_score)
        # per-frame overlay call (integrity bar + verdict):
        viz.draw_frame_overlay(frame, frame_score, segment_score, verdict,
                               fusion_weights, ai_result=None)
    """

    STATE_COLORS = {
        "COHERENT":    CLR_COHERENT,
        "SUSPICIOUS":  CLR_SUSPICIOUS,
        "INCONSISTENT": CLR_INCONSISTENT,
    }

    def draw(self, frame, track, bbox, state,
             motion_score=None, continuity_score=None):
        """
        Draw a per-track bounding box + label with forensic coloring.

        Args:
            frame: BGR numpy array (modified in-place)
            track: Track object with .track_id and .tamper_until_frame
            bbox:  (x1, y1, x2, y2)
            state: 'COHERENT' | 'SUSPICIOUS' | 'INCONSISTENT'
            motion_score: float ∈ [0,1] for the score bar
            continuity_score: float ∈ [0,1]
        """
        color = self.STATE_COLORS.get(state, CLR_WHITE)
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]

        # ── Bounding box ─────────────────────────────────────────────
        thickness = 3 if state == "INCONSISTENT" else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # ── Corner accent lines (forensic feel) ──────────────────────
        corner = 12
        for (cx, cy, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 3)

        # ── Label background ─────────────────────────────────────────
        label = f"ID:{track.track_id}  {state}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.46, 1)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame,
                      (x1, ly - th - 4), (x1 + tw + 6, ly + 2),
                      color, -1)
        cv2.putText(frame, label,
                    (x1 + 3, ly - 1), font, 0.46, CLR_BLACK, 1, cv2.LINE_AA)

        # ── Mini score bar under the box ─────────────────────────────
        if motion_score is not None:
            bar_x1, bar_y = x1, y2 + 3
            bar_w = max(1, x2 - x1)
            bar_h = 4
            fill = int(bar_w * max(0.0, min(1.0, motion_score)))
            # background
            cv2.rectangle(frame,
                          (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h),
                          (40, 40, 40), -1)
            # filled portion
            cv2.rectangle(frame,
                          (bar_x1, bar_y), (bar_x1 + fill, bar_y + bar_h),
                          color, -1)

        # ── Tamper latch indicator ────────────────────────────────────
        from coherex.config import CONFIG
        # (track.tamper_until_frame is set externally by the pipeline)
        # We can check if state is INCONSISTENT as proxy
        if state == "INCONSISTENT":
            warn = "! TAMPER"
            (ww, wh), _ = cv2.getTextSize(warn, font, 0.42, 1)
            wx = x2 - ww - 6
            wy = y2 - 6
            if wx > x1 and wy > y1:
                cv2.rectangle(frame,
                              (wx - 2, wy - wh - 2), (wx + ww + 2, wy + 2),
                              CLR_INCONSISTENT, -1)
                cv2.putText(frame, warn,
                            (wx, wy), font, 0.42, CLR_WHITE, 1, cv2.LINE_AA)

    def draw_frame_overlay(self, frame, frame_score: float,
                           segment_score: float, verdict: str,
                           fusion_weights: tuple = (0.7, 0.3, 0.0),
                           ai_result: dict = None):
        """
        Draw the per-frame global integrity overlay:
          - Integrity score + verdict badge (top-left corner)
          - Segment score
          - Horizontal integrity bar
          - AI classifier verdict (if available)
          - Red warning stripe at bottom when COMPROMISED
        """
        h, w = frame.shape[:2]
        font  = cv2.FONT_HERSHEY_SIMPLEX
        font2 = cv2.FONT_HERSHEY_DUPLEX

        # ── Verdict color ─────────────────────────────────────────────
        if verdict == "HIGH":
            vc = CLR_COHERENT
        elif verdict == "MODERATE":
            vc = CLR_SUSPICIOUS
        else:
            vc = CLR_INCONSISTENT

        # ── Semi-transparent top banner ──────────────────────────────
        overlay  = frame.copy()
        banner_h = 58
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (8, 10, 20), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # ── Integrity score ───────────────────────────────────────────
        score_txt = f"INTEGRITY  {frame_score:.3f}"
        cv2.putText(frame, score_txt,
                    (12, 24), font2, 0.58, vc, 1, cv2.LINE_AA)

        # ── Verdict badge ─────────────────────────────────────────────
        badge = f"[ {verdict} ]"
        (bw, bh), _ = cv2.getTextSize(badge, font2, 0.56, 2)
        cv2.putText(frame, badge,
                    (12, 50), font2, 0.56, vc, 2, cv2.LINE_AA)

        # ── Segment score ─────────────────────────────────────────────
        seg_txt = f"SEG: {segment_score:.3f}"
        cv2.putText(frame, seg_txt,
                    (180, 50), font, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Horizontal integrity bar (right side of banner) ───────────
        bar_x0 = 330
        bar_x1 = w - 12
        bar_y  = 28
        bar_h  = 12
        bar_w  = max(1, bar_x1 - bar_x0)
        fill   = int(bar_w * max(0.0, min(1.0, frame_score)))
        cv2.rectangle(frame, (bar_x0, bar_y - bar_h),
                      (bar_x1, bar_y), (40, 40, 50), -1)
        cv2.rectangle(frame, (bar_x0, bar_y - bar_h),
                      (bar_x0 + fill, bar_y), vc, -1)
        # threshold markers
        high_x = bar_x0 + int(bar_w * 0.85)
        mod_x  = bar_x0 + int(bar_w * 0.60)
        cv2.line(frame, (high_x, bar_y - bar_h - 2),
                 (high_x, bar_y + 2), CLR_COHERENT, 1)
        cv2.line(frame, (mod_x, bar_y - bar_h - 2),
                 (mod_x, bar_y + 2), CLR_SUSPICIOUS, 1)

        # ── Weights legend (right-aligned) ────────────────────────────
        wm, wc, wr = fusion_weights
        wt = f"W: mot={wm:.1f} cnt={wc:.1f} crd={wr:.1f}"
        (ww_, _), _ = cv2.getTextSize(wt, font, 0.37, 1)
        cv2.putText(frame, wt,
                    (w - ww_ - 8, 50), font, 0.37, (120, 120, 140), 1, cv2.LINE_AA)

        # ── AI classifier overlay (if available) ─────────────────────
        if ai_result and ai_result.get("available"):
            ai_conf    = ai_result.get("confidence", 0.0)
            ai_verdict = ai_result.get("verdict", "?")
            ai_color   = CLR_COHERENT if ai_conf < 0.4 else (
                         CLR_SUSPICIOUS if ai_conf < 0.65 else CLR_INCONSISTENT)
            ai_txt = f"AI: {ai_verdict}  {ai_conf:.0%}"
            cv2.putText(frame, ai_txt,
                        (bar_x0, 24), font, 0.44, ai_color, 1, cv2.LINE_AA)

        # ── Red warning stripe at bottom when COMPROMISED ─────────────
        if verdict == "COMPROMISED":
            stripe_h = 6
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - stripe_h),
                          (w, h), CLR_INCONSISTENT, -1)
            cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

            # Flashing "TAMPERED" text in bottom-center
            warn2 = "!! MOTION DISCONTINUITY / TAMPER DETECTED !!"
            (tw2, th2), _ = cv2.getTextSize(warn2, font, 0.52, 2)
            tx = max(0, (w - tw2) // 2)
            ty = h - stripe_h - 8
            cv2.putText(frame, warn2,
                        (tx, ty), font, 0.52, CLR_INCONSISTENT, 2, cv2.LINE_AA)
        elif verdict == "MODERATE":
            # Amber bottom stripe
            stripe_h = 4
            overlay3 = frame.copy()
            cv2.rectangle(overlay3, (0, h - stripe_h),
                          (w, h), CLR_SUSPICIOUS, -1)
            cv2.addWeighted(overlay3, 0.45, frame, 0.55, 0, frame)
