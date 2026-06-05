# scripts/train_hybrid.py
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

FEATURES = [
    "motion_min",
    "motion_mean",
    "min_integrity",
    "volatility",
    "anomaly_density",
    "perc_below_06",
    "max_mcv",
    "flow_var",
    "ssim_mean",
    "accel_kurt"
]

def main():
    csv_path = os.path.join(ROOT, "data", "evaluation", "results_test.csv")
    df = pd.read_csv(csv_path).dropna()
    
    # Preprocessing
    df["max_mcv_log"] = np.log1p(df["max_mcv"].clip(lower=0))
    feats = [f for f in FEATURES if f != "max_mcv"] + ["max_mcv_log"]
    
    X = df[feats]
    y = df["label"]
    
    print(f"[HYBRID] Training on {len(y)} samples with {len(feats)} features...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    
    model_kwargs = {
        "n_estimators": 300,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        clf = RandomForestClassifier(**model_kwargs)
        clf.fit(X_tr, y_tr)
        
        probas = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probas)
        fold_aucs.append(auc)
        print(f"      Fold {fold}: AUC = {auc:.4f}")

    print(f"\n[HYBRID] MEAN AUC: {np.mean(fold_aucs):.4f} \u00b1 {np.std(fold_aucs):.4f}")

if __name__ == "__main__":
    main()
