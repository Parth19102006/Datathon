import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def get_class_weights(y):
    classes = np.unique(y)
    weights = np.zeros_like(y, dtype=np.float64)
    total_samples = len(y)
    for c in classes:
        n_samples_c = np.sum(y == c)
        w = total_samples / (len(classes) * n_samples_c)
        weights[y == c] = w
    return weights

def main():
    print("Loading data...")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    
    id_col = "id"
    target_col = "target"
    feature_cols = [c for c in df_train.columns if c not in [id_col, target_col]]
    
    X = df_train[feature_cols]
    y_encoded = df_train[target_col].values - 1
    X_test = df_test[feature_cols]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    print("Computing sample weights...")
    sample_weight = get_class_weights(y_train)
    
    print("Training XGBoost...")
    # These are sensible parameters for a diverse XGBoost model
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 7,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 800,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4
    }
    
    t0 = time.time()
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_train_imp, y_train, sample_weight=sample_weight)
    print(f"XGBoost training took {time.time()-t0:.2f} seconds.")
    
    print("Predicting probabilities...")
    xgb_val_proba = xgb_model.predict_proba(X_val_imp)
    xgb_test_proba = xgb_model.predict_proba(X_test_imp)
    
    np.save("val_proba_xgb_best.npy", xgb_val_proba)
    np.save("test_proba_xgb_best.npy", xgb_test_proba)
    
    xgb_preds = np.argmax(xgb_val_proba, axis=1)
    xgb_f1 = f1_score(y_val, xgb_preds, average="macro")
    print(f"XGBoost Standalone Macro-F1: {xgb_f1:.6f}")
    
    print("Loading current LGBM probabilities for comparison...")
    lgbm_val_proba = np.load("val_proba_lgbm_best.npy")
    lgbm_preds = np.argmax(lgbm_val_proba, axis=1)
    
    agreement = np.mean(xgb_preds == lgbm_preds)
    print(f"Agreement with LGBM predictions: {agreement:.4f}")
    
    print("\nBlending with current LGBM probabilities:")
    blend_weights = [
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6)
    ]
    
    results = {}
    best_f1 = 0
    best_weights = None
    
    for wl, wx in blend_weights:
        p = wl * lgbm_val_proba + wx * xgb_val_proba
        f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
        print(f"LGBM {wl*100:.0f} / XGB {wx*100:.0f} -> Macro-F1: {f1:.6f}")
        results[f"LGBM_{wl}_XGB_{wx}"] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (wl, wx)
            
    print(f"\nBest Blend: LGBM {best_weights[0]} / XGB {best_weights[1]} (F1: {best_f1:.6f})")
    
    with open("phase3_report.json", "w") as f:
        json.dump({
            "xgb_standalone_f1": xgb_f1,
            "agreement": agreement,
            "blend_results": results,
            "best_blend": best_weights,
            "best_blend_f1": best_f1
        }, f, indent=2)

if __name__ == "__main__":
    main()
