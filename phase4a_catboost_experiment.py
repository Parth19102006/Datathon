import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier

def main():
    print("=== Phase 4A: CatBoost Baseline & Ensemble Experiment ===")
    
    # 1. Load Data
    print("Loading data...")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    
    id_col = "id"
    target_col = "target"
    
    feature_cols = [c for c in df_train.columns if c not in [id_col, target_col]]
    
    X = df_train[feature_cols]
    y = df_train[target_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_order = le.classes_.tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # 2. Train CatBoost
    print("\nTraining CatBoost Baseline...")
    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "random_seed": 42,
        "verbose": 100,
        "thread_count": -1,
        "task_type": "CPU",
    }
    
    cb_model = CatBoostClassifier(**cb_params)
    
    t0 = time.time()
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, use_best_model=True)
    print(f"CatBoost fitted in {time.time() - t0:.2f}s")
    
    # 3. Predict & Evaluate CatBoost
    cb_val_proba = cb_model.predict_proba(X_val)
    np.save("cb_val_proba.npy", cb_val_proba)
    print("Saved cb_val_proba.npy")
    
    cb_val_preds = np.argmax(cb_val_proba, axis=1)
    cb_macro_f1 = f1_score(y_val, cb_val_preds, average="macro")
    
    print(f"\nCatBoost Standalone Macro F1: {cb_macro_f1:.6f}")
    print("Per-class F1:")
    per_class_f1 = f1_score(y_val, cb_val_preds, average=None)
    for cls_idx, score in enumerate(per_class_f1):
        print(f"  {class_order[cls_idx]}: {score:.4f}")
        
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, cb_val_preds))
    
    print("\nClass Prediction Distribution:")
    unique, counts = np.unique(cb_val_preds, return_counts=True)
    dist = dict(zip([class_order[u] for u in unique], counts))
    print(dist)
    
    # 4. Load Existing Probabilities
    print("\nLoading LightGBM and XGBoost probabilities...")
    lgb_val_proba = np.load("lgbm_val_proba.npy")
    xgb_val_proba = np.load("xgb_val_proba.npy")
    
    # Evaluate 86:14 LGB/XGB baseline blend
    w_lgb_base = 0.86
    w_xgb_base = 0.14
    base_blend_proba = w_lgb_base * lgb_val_proba + w_xgb_base * xgb_val_proba
    base_blend_preds = np.argmax(base_blend_proba, axis=1)
    base_blend_f1 = f1_score(y_val, base_blend_preds, average="macro")
    print(f"Existing LGBM/XGBoost Blend (86:14) Macro F1: {base_blend_f1:.6f}")
    
    # 5. Blend Search
    print("\nSearching for best blends...")
    weights = np.linspace(0, 1.0, 101)
    
    # LGBM + CB
    best_lgb_cb_f1 = 0
    best_lgb_cb_w = 0
    for w in weights:
        p = w * lgb_val_proba + (1-w) * cb_val_proba
        f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
        if f1 > best_lgb_cb_f1:
            best_lgb_cb_f1 = f1
            best_lgb_cb_w = w
            
    # XGB + CB
    best_xgb_cb_f1 = 0
    best_xgb_cb_w = 0
    for w in weights:
        p = w * xgb_val_proba + (1-w) * cb_val_proba
        f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
        if f1 > best_xgb_cb_f1:
            best_xgb_cb_f1 = f1
            best_xgb_cb_w = w
            
    # LGBM + XGB + CB (3-way)
    best_3way_f1 = 0
    best_3way_weights = (0,0,0)
    
    for w1 in np.linspace(0, 1.0, 51):
        for w2 in np.linspace(0, 1.0 - w1, 51):
            w3 = 1.0 - w1 - w2
            p = w1 * lgb_val_proba + w2 * xgb_val_proba + w3 * cb_val_proba
            f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
            if f1 > best_3way_f1:
                best_3way_f1 = f1
                best_3way_weights = (w1, w2, w3)
                
    print(f"\nBest LGBM + CB Macro F1:  {best_lgb_cb_f1:.6f} (LGB={best_lgb_cb_w:.2f}, CB={1-best_lgb_cb_w:.2f})")
    print(f"Best XGB + CB Macro F1:   {best_xgb_cb_f1:.6f} (XGB={best_xgb_cb_w:.2f}, CB={1-best_xgb_cb_w:.2f})")
    
    w1, w2, w3 = best_3way_weights
    print(f"Best 3-way Blend Macro F1: {best_3way_f1:.6f} (LGB={w1:.2f}, XGB={w2:.2f}, CB={w3:.2f})")
    
    improvement = best_3way_f1 - base_blend_f1
    print(f"\nImprovement over 0.8041 Blend: {improvement:.6f}")
    
    if improvement > 1e-5:
        print("Conclusion: CatBoost IMPROVES the blend and adds complementary signal.")
        best_blend_proba = w1 * lgb_val_proba + w2 * xgb_val_proba + w3 * cb_val_proba
        np.save("best_3way_val_proba.npy", best_blend_proba)
        
        print("Generating test probabilities for CatBoost...")
        X_test = df_test[[c for c in df_test.columns if c != "id"]]
        cb_test_proba = cb_model.predict_proba(X_test)
        np.save("cb_test_proba.npy", cb_test_proba)
        
        lgb_test_proba = np.load("lgbm_test_proba.npy")
        xgb_test_proba = np.load("xgb_test_proba.npy")
        best_3way_test_proba = w1 * lgb_test_proba + w2 * xgb_test_proba + w3 * cb_test_proba
        np.save("best_3way_test_proba.npy", best_3way_test_proba)
        print("Saved cb_test_proba.npy and best_3way_test_proba.npy")
        
        res_dict = {
            "lgbm_weight": w1,
            "xgb_weight": w2,
            "cb_weight": w3,
            "best_val_f1": best_3way_f1,
            "improvement": improvement
        }
        with open("phase4a_best_blend_weights.json", "w") as f:
            json.dump(res_dict, f, indent=2)
    else:
        print("Conclusion: CatBoost DOES NOT improve the blend and should not be included.")
        
    print("Done.")

if __name__ == "__main__":
    main()
