# CoheRex Evaluation Report

> **Generated**: *[auto-fill after evaluation runs]*
> **System**: CoheRex-Integrity v1.0

---

## 1. Dataset Description

| Category | Count | Source |
|----------|-------|--------|
| Authentic | 20 | Extracted as non-overlapping 10-second segments from `sample.mp4` and `sample1.mp4` |
| Tampered  | 20 | Derived from the same sources using 5 synthetic tampering types |
| **Total** | **40** | |

---

## 2. Tampering Types

| # | Type                | Description |
|---|---------------------|-------------|
| 1 | Frame Deletion      | 5–20 consecutive frames removed at a random interior point |
| 2 | Frame Duplication   | A 1.5-second segment repeated back-to-back |
| 3 | Speed Manipulation  | 2-second window accelerated by dropping every 2nd frame |
| 4 | Clip Splice         | 2–3 seconds from the other source video inserted |
| 5 | Reverse Segment     | 1.5-second segment played in reverse (temporal inversion) |

Each type is applied with **4 random seeds** → 5 × 4 = 20 tampered clips.

---

## 3. Evaluation Metric

**One scalar per video**: mean segment integrity score (window=150 frames).

`tamper_score = 1 - integrity_score` (so higher = more tampered, aligning with ROC convention).

---

## 4. ROC Curve

![ROC Curve](data/evaluation/roc_curve.png)

| Metric | Value |
|--------|-------|
| **AUC** | *[see results]* |
| Interpretation | *[see below]* |

---

## 5. Confusion Matrix (threshold = 0.60)

![Confusion Matrix](data/evaluation/confusion_matrix.png)

| Metric    | Value |
|-----------|-------|
| Accuracy  | *[see results]* |
| Precision | *[see results]* |
| Recall    | *[see results]* |
| F1-score  | *[see results]* |

---

## 6. Sensitivity Analysis (Window Size)

![Sensitivity](data/evaluation/sensitivity_auc_comparison.png)

| Window (frames) | AUC |
|-----------------|-----|
| 100             | *[see results]* |
| 150             | *[see results]* |
| 300             | *[see results]* |

---

## 7. Ablation Study

![Ablation ROC](data/evaluation/ablation_roc_comparison.png)

| Agent Configuration    | AUC |
|------------------------|-----|
| Motion only            | *[see results]* |
| Motion + Continuity    | *[see results]* |
| Motion + Crowd         | *[see results]* |
| Full fusion            | *[see results]* |

---

## 8. Interpretation

> **AUC > 0.90** → Strong discriminative power  
> **AUC 0.80–0.90** → Good  
> **AUC 0.70–0.80** → Weak but usable  
> **AUC < 0.70** → Redesign required

---

## 9. Execution Commands

```bash
# Step 1 — Build dataset
python scripts/create_tampered_videos.py

# Step 2 — Run batch evaluation
python scripts/evaluate_dataset.py

# Step 3 — ROC + confusion matrix
python scripts/compute_roc.py

# Step 4 — Sensitivity analysis
python scripts/sensitivity_analysis.py

# Step 5 — Ablation study
python scripts/ablation_study.py
```
