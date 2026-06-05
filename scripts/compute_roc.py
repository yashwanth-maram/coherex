"""
compute_roc.py
==============
Loads data/evaluation/results.csv and produces:

  1. ROC curve plot  + AUC  (saved to data/evaluation/roc_curve.png)
  2. Confusion matrix at the current threshold (0.60)
  3. Accuracy, Precision, Recall, F1

Usage:
    python scripts/compute_roc.py [--csv path] [--threshold 0.60]

Convention:
    integrity_score → LOW means tampered.
    ROC requires "higher = more positive", so:
        tamper_score = 1 - integrity_score
"""

import os
import sys
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix
except ImportError:
    print("[ERROR] scikit-learn is required: pip install scikit-learn")
    sys.exit(1)


DEFAULT_CSV       = os.path.join(ROOT, "data", "evaluation", "results.csv")
DEFAULT_THRESHOLD = 0.60
PLOT_DIR          = os.path.join(ROOT, "data", "evaluation")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: str):
    """Return a pandas DataFrame for easier multi-metric handling."""
    return pd.read_csv(path)


def compute_metrics(labels, predictions):
    """Compute TP/FP/TN/FN and derived metrics."""
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    n = tp + tn + fp + fn
    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return dict(TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1)


def classify_at_threshold(integrity_scores, threshold):
    """
    Predict label (0/1) using integrity threshold.
    integrity < threshold  → tampered (1)
    integrity >= threshold → authentic (0)
    """
    return [1 if s < threshold else 0 for s in integrity_scores]


# ─── Plot helpers ──────────────────────────────────────────────────────────────

def plot_roc(fpr, tpr, roc_auc, out_path, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--", lw=1,
            label="Random classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve saved → {out_path}")


def plot_confusion_matrix(cm_dict, out_path, threshold):
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array([[cm_dict["TN"], cm_dict["FP"]],
                        [cm_dict["FN"], cm_dict["TP"]]])
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, str(val), ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if val > matrix.max() * 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Authentic", "Pred: Tampered"], fontsize=11)
    ax.set_yticklabels(["True: Authentic", "True: Tampered"], fontsize=11)
    ax.set_title(f"Confusion Matrix  (threshold={threshold})", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CoheRex ROC + Metrics")
    p.add_argument("--csv",       type=str,   default=DEFAULT_CSV)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--tag",       type=str,   default="",
                   help="Optional tag appended to output filenames (e.g. 'w100')")
    return p.parse_args()


def run(csv_path, threshold=DEFAULT_THRESHOLD, tag=""):
    """
    Enhanced logic: Computes ROC for multiple metrics and saves a comparison plot.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return None

    df = load_csv(csv_path)
    if df.empty:
        print("[ERROR] CSV is empty")
        return None

    # Filter out failed runs
    df = df[df['integrity_score'] >= 0]
    labels = df['label'].values

    metrics = [
        "integrity_score",
        "min_integrity",
        "motion_mean",
        "continuity_mean",
        "crowd_mean",
        "perc_below_06",
        "anomaly_density",
        "max_mcv",
        "burst_length",
        "ssim_mean",
        "flow_var",
        "accel_kurt"
    ]

    results = {}
    tag_suffix = f"_{tag}" if tag else ""

    plt.figure(figsize=(10, 8))
    
    for m in metrics:
        if m not in df.columns:
            continue
            
        scores = df[m].values
        # Convention: higher tamper_score means more likely tampered.
        # For our integrity agents: 1.0 (auth) -> 0.0 (tamp)
        # So tamper_scores = 1.0 - integrity_scores
        # Convention: higher tamper_score means more likely tampered.
        if m in ["perc_below_06", "anomaly_density", "max_mcv", "burst_length", "flow_var", "accel_kurt"]:
            t_scores = scores
        elif m == "ssim_mean":
            t_scores = 1.0 - scores
        else:
            t_scores = 1.0 - scores

        fpr, tpr, _ = roc_curve(labels, t_scores)
        roc_auc = auc(fpr, tpr)
        
        results[m] = {"auc": roc_auc, "fpr": fpr, "tpr": tpr}
        plt.plot(fpr, tpr, lw=2, label=f"{m:15s} (AUC={roc_auc:.4f})")
        print(f"  [METRIC] {m:15s} AUC: {roc_auc:.4f}")
    
    plt.plot([0, 1], [0, 1], color="#aaaaaa", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"CoheRex ROC Comparison {tag}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    comp_path = os.path.join(PLOT_DIR, f"roc_comparison{tag_suffix}.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"  ROC comparison saved: {comp_path}")

    # Generate Histograms for key metrics to analyze distribution overlap
    import seaborn as sns
    
    # Statistical Summary (Step 6 requirement)
    if "max_mcv" in df.columns:
        auth_mcv = df[df["label"] == 0]["max_mcv"]
        hard_mcv = df[df["label"] == 1]["max_mcv"]
        print(f"\n[STATS] Mean(max_mcv) Authentic : {auth_mcv.mean():.4f}")
        print(f"[STATS] Mean(max_mcv) Tampered  : {hard_mcv.mean():.4f}")
        if hard_mcv.mean() > auth_mcv.mean():
            print(f"[STATS] Separation Check: PASS (Tampered Mean > Authentic Mean)")
        else:
            print(f"[STATS] Separation Check: FAIL")

    for m in ["min_integrity", "anomaly_density", "max_mcv", "burst_length", "ssim_mean", "flow_var", "accel_kurt"]:
        if m not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=m, hue="label", element="step", common_norm=False, kde=True)
        plt.title(f"Burst Anomaly Distribution: {m} (Auth=0, Tamp=1)")
        plt.grid(True, alpha=0.2)
        dist_path = os.path.join(PLOT_DIR, f"dist_{m}{tag_suffix}.png")
        plt.savefig(dist_path, dpi=150)
        plt.close()
        print(f"  [PLOT] Saved distribution for {m}: {dist_path}")

    # For backward compatibility / standard reporting, return integrity_score metrics
    main_m = "integrity_score"
    main_auc = results[main_m]["auc"]
    
    # Simple threshold classification for integrity_score only
    preds = [1 if s < threshold else 0 for s in df['integrity_score'].values]
    metrics_dict = compute_metrics(labels.tolist(), preds)

    return {
        "roc_auc": main_auc,
        "metrics": metrics_dict,
        "results": results,
        "n_samples": len(df),
        "n_authentic": int((labels == 0).sum()),
        "n_tampered":  int((labels == 1).sum())
    }


def print_report(result, threshold):
    if result is None:
        return
    cm = result["metrics"]
    print()
    print("=" * 60)
    print("  COHEREX EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples   : {result['n_samples']}  "
          f"(authentic={result['n_authentic']}, tampered={result['n_tampered']})")
    print()
    print(f"  AUC       : {result['roc_auc']:.4f}")
    if result["roc_auc"] > 0.90:
        tier = "★★★  Strong discriminative power"
    elif result["roc_auc"] > 0.80:
        tier = "★★   Good"
    elif result["roc_auc"] > 0.70:
        tier = "★    Weak but usable"
    else:
        tier = "✗    Redesign required"
    print(f"           → {tier}")
    print()
    print(f"  Threshold : {threshold}")
    print(f"  TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")
    print(f"  Accuracy  : {cm['accuracy']:.4f}")
    print(f"  Precision : {cm['precision']:.4f}")
    print(f"  Recall    : {cm['recall']:.4f}")
    print(f"  F1-score  : {cm['f1']:.4f}")
    print("=" * 60)


def main():
    args = parse_args()
    result = run(args.csv, args.threshold, args.tag)
    print_report(result, args.threshold)


if __name__ == "__main__":
    main()
