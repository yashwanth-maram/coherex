"""
train_logistic_roc.py
=====================
Implements Step 6 of the CoheRex 0.75 AUC Roadmap.
Trains a Logistic Regression model on burst-anomaly features to find
the optimal linear separation between Authentic and Tampered classes.

Features:
- anomaly_density
- max_mcv
- min_integrity
- continuity_mean
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def run(csv_path, tag="Baseline"):
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter out failed rows
    df = df[df['integrity_score'] >= 0].copy()
    
    features = ["anomaly_density", "max_mcv", "min_integrity", "continuity_mean", "burst_length", "ssim_mean", "flow_var", "accel_kurt"]
    X = df[features]
    y = df["label"]
    
    print(f"\n[TRAIN] Training Logistic Regression on {len(df)} samples ({tag})")
    print(f"        Features: {features}")
    
    # Train model (using L2 regularization by default)
    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X, y)
    
    # Predict probabilities for ROC
    y_probs = clf.predict_proba(X)[:, 1]
    
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n[RESULTS] Logistic Fusion AUC: {roc_auc:.4f}")
    
    # Feature Importance (Coefficients)
    importance = pd.DataFrame({
        'feature': features,
        'coef': clf.coef_[0]
    }).sort_values(by='coef', ascending=False)
    
    print("\n[IMPORTANCE] Feature Coefficients:")
    print(importance)
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Logistic Fusion (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: Logistic Forensic Fusion ({tag})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_plot = csv_path.replace(".csv", "_logistic_roc.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"\n[PLOT] Saved fusion ROC → {out_plot}")
    
    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--tag", type=str, default="Baseline")
    args = parser.parse_args()
    
    run(args.csv, args.tag)

if __name__ == "__main__":
    main()
