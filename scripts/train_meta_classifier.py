# scripts/train_meta_classifier.py
"""
PERFECT EXECUTION: Random Forest Meta-Classifier Training
==========================================================
Optimizer for CoheRex classification performance (AUC >= 0.75).
Follows strict instructions: limited features, specific RF params,
and stratified 5-fold cross-validation.

Usage:
    python scripts/train_meta_classifier.py --csv data/evaluation/results_test.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, roc_curve,
    classification_report, accuracy_score,
)

# Ensure coherex is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

# ─── 1️⃣ Feature Selection (STRICT) ───────────────────────────────────────────
SELECTED_FEATURES = [
    "motion_min",
    "motion_mean",
    "min_integrity",
    "volatility",
    "anomaly_density",
    "perc_below_06",
]
OPTIONAL_FEATURES = ["max_mcv"]

def parse_args():
    p = argparse.ArgumentParser(description="CoheRex Meta-Classifier Trainer")
    p.add_argument(
        "--csv",
        default=os.path.join(ROOT, "data", "evaluation", "results_test.csv"),
        help="Path to evaluation results CSV",
    )
    p.add_argument(
        "--output",
        default=os.path.join(ROOT, "data", "models", "meta_classifier.pkl"),
        help="Where to save the trained model bundle",
    )
    p.add_argument("--use_max_mcv", action="store_true", default=True,
                   help="Include log(max_mcv) in features")
    return p.parse_args()

def prepare_data(df: pd.DataFrame, use_max_mcv=True):
    # Select available features
    features = SELECTED_FEATURES.copy()
    if use_max_mcv and "max_mcv" in df.columns:
        # 2️⃣ Preprocessing: Apply log(1 + x) ONLY to max_mcv
        df["max_mcv_log"] = np.log1p(df["max_mcv"].clip(lower=0))
        features.append("max_mcv_log")
        print("[PREPROC] Applied log(1+x) to max_mcv -> max_mcv_log")
    
    X = df[features]
    y = df["label"]
    
    # 5️⃣ Sanity Checks Before Training
    print("\n[SANITY] Class Distribution:")
    print(y.value_counts())
    
    print("\n[SANITY] Feature Correlation Matrix:")
    corr = X.corr()
    print(corr.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Remove any feature with correlation > 0.9
    to_drop = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                col_i = corr.columns[i]
                col_j = corr.columns[j]
                print(f"[SANITY] HIGH CORRELATION DETECTED: {col_i} vs {col_j} ({corr.iloc[i,j]:.4f})")
                if col_i not in to_drop:
                    to_drop.append(col_i)
    
    if to_drop:
        print(f"[SANITY] Dropping redundant features: {to_drop}")
        X = X.drop(columns=to_drop)
        features = [f for f in features if f not in to_drop]
        
    return X, y, features

def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)
        
    df = pd.read_csv(args.csv).dropna()
    X_raw, y, feature_names = prepare_data(df, args.use_max_mcv)
    X = X_raw.values
    
    # 3️⃣ Model Configuration
    model_kwargs = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }
    
    # 4️⃣ Evaluation Protocol (StratifiedKFold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    conf_matrices = []
    
    print(f"\n[TRAIN] Starting 5-Fold CV on {len(y)} samples...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = RandomForestClassifier(**model_kwargs)
        clf.fit(X_tr, y_tr)
        
        probas = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probas)
        fold_aucs.append(auc)
        
        # Confusion matrix for this fold
        preds = clf.predict(X_val)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        conf_matrices.append(cm)
        
        # ROC curve tracking
        fpr, tpr, _ = roc_curve(y_val, probas)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        print(f"      Fold {fold}: AUC = {auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    # 8️⃣ Final Output Format
    print(f"\n{'='*40}")
    print(f"1. MEAN AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'='*40}")
    
    # Train final model on ALL data
    final_clf = RandomForestClassifier(**model_kwargs)
    final_clf.fit(X, y)
    
    # 2. Ranked Feature Importances
    importances = final_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n2. Ranked Feature Importances:")
    print(f"   {'Feature':<20} | {'Importance':>10}")
    print(f"   {'-'*33}")
    for i in indices:
        print(f"   {feature_names[i]:<20} | {importances[i]:>10.4f}")
        
    # 3. Confusion Matrix (Average over folds)
    avg_cm = np.mean(conf_matrices, axis=0)
    print("\n3. Confusion Matrix (Average over folds):")
    print(avg_cm)
    
    # 4. Interpretation
    top_feature = feature_names[indices[0]]
    print(f"\n4. Interpretation: {top_feature} appears to dominate the classification signal.")

    # ── OFFICIAL METRICS BLOCK ─────────────────────────────────────────────
    # Use cross_val_predict to get OOF predictions across ALL samples.
    # Same model kwargs + same SKF strategy — no data leakage.
    print(f"\n{'='*60}")
    print("  OFFICIAL METRICS BLOCK (cross_val_predict, OOF)")
    print(f"{'='*60}")

    skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_clf  = RandomForestClassifier(**model_kwargs)

    # Hard class predictions (threshold = 0.5)
    y_pred_oof  = cross_val_predict(oof_clf, X, y, cv=skf_oof, method="predict")
    # Probability for ROC / threshold sweep
    y_proba_oof = cross_val_predict(oof_clf, X, y, cv=skf_oof, method="predict_proba")[:, 1]

    # ── Confusion matrix at default 0.5 threshold ──────────────────────────
    cm_oof = confusion_matrix(y, y_pred_oof, labels=[0, 1])
    tn, fp, fn, tp = cm_oof.ravel()
    n = tn + fp + fn + tp

    accuracy    = (tp + tn) / n
    precision   = tp / (tp + fp)   if (tp + fp)   else 0.0
    recall      = tp / (tp + fn)   if (tp + fn)   else 0.0  # sensitivity
    specificity = tn / (tn + fp)   if (tn + fp)   else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) else 0.0)
    oof_auc     = roc_auc_score(y, y_proba_oof)

    print(f"\n  Dataset    : {n} samples  "
          f"(authentic={int((y==0).sum())}, tampered={int((y==1).sum())})")
    print(f"  Threshold  : 0.50  (hard predict)")
    print()
    print(f"  Confusion Matrix:")
    print(f"                   Pred Authentic   Pred Tampered")
    print(f"  True Authentic       TN={tn:<5}         FP={fp}")
    print(f"  True Tampered        FN={fn:<5}         TP={tp}")
    print()
    print(f"  AUC          : {oof_auc:.4f}")
    print(f"  Accuracy     : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision    : {precision:.4f}  (of tampered calls, how many are real)")
    print(f"  Recall       : {recall:.4f}  (of real tampered, how many found)")
    print(f"  Specificity  : {specificity:.4f}  (of real authentic, how many kept)")
    print(f"  F1-score     : {f1:.4f}")
    print()

    # ── sklearn classification_report ──────────────────────────────────────
    print("  Classification Report (sklearn):")
    print(classification_report(
        y, y_pred_oof,
        target_names=["Authentic (0)", "Tampered (1)"],
        digits=4,
    ))
    print(f"{'='*60}")

    # 6️⃣ After Training Checks
    if mean_auc >= 0.75:
        print("\n→ Baseline achieved (AUC >= 0.75).")
    elif mean_auc < 0.70:
        print("\n→ Separability is insufficient in selected feature space.")
    
    # ROC curve plotting
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Meta-Classifier ROC Curve (5-Fold CV)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    roc_path = args.output.replace(".pkl", "_roc.png")
    plt.savefig(roc_path)
    print(f"\n[SAVED] ROC Curve -> {roc_path}")
    
    # Save Model Bundle
    import joblib
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    bundle = {
        "model": final_clf,
        "feature_names": feature_names,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "feature_selection": "strict_baseline"
    }
    joblib.dump(bundle, args.output)
    print(f"\n[SAVED] Model bundle -> {args.output}")

if __name__ == "__main__":
    main()
