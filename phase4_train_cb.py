import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

def get_class_weights(y):
    classes = np.unique(y)
    weights = np.zeros(len(classes), dtype=np.float64)
    total_samples = len(y)
    for c in classes:
        n_samples_c = np.sum(y == c)
        w = total_samples / (len(classes) * n_samples_c)
        weights[c] = w
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
    
    # CatBoost can handle NaNs internally if passed directly, 
    # but we will use the same median imputed data for consistency in blending logic
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    print("Computing class weights...")
    class_weights = get_class_weights(y_train)
    
    print("Training CatBoost...")
    cb_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "random_seed": 42,
        "class_weights": class_weights.tolist(),
        "verbose": False,
        "task_type": "CPU"
    }
    
    t0 = time.time()
    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(X_train_imp, y_train, eval_set=(X_val_imp, y_val), early_stopping_rounds=50)
    print(f"CatBoost training took {time.time()-t0:.2f} seconds.")
    
    print("Predicting probabilities...")
    cb_val_proba = cb_model.predict_proba(X_val_imp)
    cb_test_proba = cb_model.predict_proba(X_test_imp)
    
    np.save("val_proba_cb_best.npy", cb_val_proba)
    np.save("test_proba_cb_best.npy", cb_test_proba)
    
    cb_preds = np.argmax(cb_val_proba, axis=1)
    cb_f1 = f1_score(y_val, cb_preds, average="macro")
    print(f"CatBoost Standalone Macro-F1: {cb_f1:.6f}")
    
    print("\nLoading current LGBM & XGB probabilities for comparison...")
    lgbm_val_proba = np.load("val_proba_lgbm_best.npy")
    xgb_val_proba = np.load("val_proba_xgb_best.npy")
    
    # Get the best LGBM / XGB blend from Phase 3
    with open("phase3_report.json", "r") as f:
        p3_report = json.load(f)
        
    best_lgbm_w, best_xgb_w = p3_report["best_blend"]
    print(f"Best LGBM/XGB blend is {best_lgbm_w}/{best_xgb_w}")
    
    base_blend_proba = best_lgbm_w * lgbm_val_proba + best_xgb_w * xgb_val_proba
    
    print("\nBlending CatBoost with the current LGBM/XGB blend:")
    blend_weights = [0.05, 0.1, 0.15, 0.2]
    
    results = {}
    best_f1 = p3_report["best_blend_f1"]
    best_cb_w = 0.0
    
    for cb_w in blend_weights:
        # Normalize the other weights
        remain_w = 1.0 - cb_w
        w_l = best_lgbm_w * remain_w
        w_x = best_xgb_w * remain_w
        
        p = w_l * lgbm_val_proba + w_x * xgb_val_proba + cb_w * cb_val_proba
        f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
        print(f"LGBM {w_l:.3f} / XGB {w_x:.3f} / CB {cb_w:.3f} -> Macro-F1: {f1:.6f}")
        results[f"CB_{cb_w}"] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_cb_w = cb_w
            
    print(f"\nBest Blend with CB: {best_cb_w} (F1: {best_f1:.6f})")
    
    with open("phase4_report.json", "w") as f:
        json.dump({
            "cb_standalone_f1": cb_f1,
            "blend_results": results,
            "best_cb_weight": best_cb_w,
            "best_blend_f1": best_f1
        }, f, indent=2)

if __name__ == "__main__":
    main()
