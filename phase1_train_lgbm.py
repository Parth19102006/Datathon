import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import lightgbm as lgb

def apply_log_offsets(proba, offsets):
    eps = 1e-15
    log_proba = np.log(proba + eps)
    log_proba += offsets
    exp_proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
    return exp_proba / np.sum(exp_proba, axis=1, keepdims=True)

def get_sqrt_balanced_weights(y):
    classes = np.unique(y)
    weights = np.zeros_like(y, dtype=np.float64)
    total_samples = len(y)
    for c in classes:
        n_samples_c = np.sum(y == c)
        w = np.sqrt(total_samples / (len(classes) * n_samples_c))
        weights[y == c] = w
    return weights

def main():
    print("Loading data...")
    df_train = pd.read_csv("train.csv")
    
    id_col = "id"
    target_col = "target"
    feature_cols = [c for c in df_train.columns if c not in [id_col, target_col]]
    
    X = df_train[feature_cols]
    y_encoded = df_train[target_col].values - 1  # 0-6
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    
    base_params = {
        "objective": "multiclass",
        "num_class": 7,
        "learning_rate": 0.055,
        "num_leaves": 31,
        "max_bin": 191,
        "min_child_samples": 30,
        "colsample_bytree": 0.9,
        "subsample": 0.85,
        "subsample_freq": 1,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "n_jobs": 4,
        "random_state": 42,
        "verbose": -1
    }
    
    print("Training Model 1 (balanced, 997 trees)...")
    p1 = base_params.copy()
    p1["n_estimators"] = 997
    p1["class_weight"] = "balanced"
    m1 = lgb.LGBMClassifier(**p1)
    m1.fit(X_train_imp, y_train)
    p1_val = m1.predict_proba(X_val_imp)
    
    print("Training Model 2 (sqrt-balanced, 995 trees)...")
    p2 = base_params.copy()
    p2["n_estimators"] = 995
    m2 = lgb.LGBMClassifier(**p2)
    sample_weight = get_sqrt_balanced_weights(y_train)
    m2.fit(X_train_imp, y_train, sample_weight=sample_weight)
    p2_val = m2.predict_proba(X_val_imp)
    
    val_proba = 0.5 * p1_val + 0.5 * p2_val
    np.save("val_proba_lgbm_best.npy", val_proba)
    np.save("y_val_exact.npy", y_val)
    
    offsets = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
    mod_proba = apply_log_offsets(val_proba, offsets)
    preds = np.argmax(mod_proba, axis=1)
    
    f1 = f1_score(y_val, preds, average="macro")
    print(f"Validation Macro F1 (with offsets): {f1:.6f}")
    
    # Generate Phase 1 report
    print("Generating Phase 1 report...")
    p, r, f, s = precision_recall_fscore_support(y_val, preds)
    
    report = f"# Phase 1: Validation Error Analysis\n\n"
    report += f"**Overall Macro-F1**: {f1:.6f}\n\n"
    
    report += "## Per-Class Metrics\n"
    report += "| Class | Precision | Recall | F1 | Support |\n"
    report += "|-------|-----------|--------|----|---------|\n"
    for i in range(7):
        report += f"| {i+1} | {p[i]:.4f} | {r[i]:.4f} | {f[i]:.4f} | {s[i]} |\n"
        
    report += "\n## Confusion Matrix\n"
    cm = confusion_matrix(y_val, preds)
    for i in range(7):
        report += str(cm[i]) + "\n"
        
    with open("phase1_report.md", "w") as f_out:
        f_out.write(report)
        
if __name__ == "__main__":
    main()
