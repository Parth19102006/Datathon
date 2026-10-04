import numpy as np
import json
import pandas as pd
from sklearn.metrics import f1_score

def get_macro_f1(y_true, proba):
    preds_idx = np.argmax(proba, axis=1)
    return f1_score(y_true, preds_idx, average='macro')

def main():
    # Load artifacts
    lgbm_val_proba = np.load("lgbm_val_proba.npy")
    xgb_val_proba = np.load("xgb_val_proba.npy")
    y_val = np.load("y_val.npy")
    
    with open("class_order.json", "r") as f:
        class_order = json.load(f)
        
    lgbm_score = get_macro_f1(y_val, lgbm_val_proba)
    xgb_score = get_macro_f1(y_val, xgb_val_proba)
    
    print(f"LightGBM = {lgbm_score:.6f}")
    print(f"XGBoost  = {xgb_score:.6f}")
    
    # 50:50 Ensemble
    ensemble_50_50 = 0.5 * lgbm_val_proba + 0.5 * xgb_val_proba
    score_50_50 = get_macro_f1(y_val, ensemble_50_50)
    print(f"50:50 Ensemble = {score_50_50:.6f}")
    print(f"Gain/Loss      = {score_50_50 - lgbm_score:.6f}\n")
    
    # Full Weight Search
    print("LGBM Weight | XGB Weight | Macro F1 | Gain vs LGBM")
    print("------------|------------|----------|-------------")
    
    best_score = lgbm_score
    best_lgbm_weight = 1.0
    best_xgb_weight = 0.0
    
    results = []
    
    for lgbm_w in np.arange(0.0, 1.01, 0.1):
        lgbm_w = round(lgbm_w, 1)
        xgb_w = round(1.0 - lgbm_w, 1)
        
        blend_proba = (lgbm_w * lgbm_val_proba) + (xgb_w * xgb_val_proba)
        score = get_macro_f1(y_val, blend_proba)
        gain = score - lgbm_score
        
        print(f"{lgbm_w:11.1f} | {xgb_w:10.1f} | {score:8.6f} | {gain:11.6f}")
        results.append((lgbm_w, xgb_w, score, gain))
        
        if score > best_score:
            best_score = score
            best_lgbm_weight = lgbm_w
            best_xgb_weight = xgb_w
            
    print("\n")
    
    # Fine Search
    if best_score > lgbm_score:
        print("--- EXPERIMENT 3 - FINE SEARCH ---")
        start_w = max(0.0, best_lgbm_weight - 0.09)
        end_w = min(1.0, best_lgbm_weight + 0.09)
        
        for lgbm_w in np.arange(start_w, end_w + 0.001, 0.01):
            lgbm_w = round(lgbm_w, 2)
            xgb_w = round(1.0 - lgbm_w, 2)
            
            blend_proba = (lgbm_w * lgbm_val_proba) + (xgb_w * xgb_val_proba)
            score = get_macro_f1(y_val, blend_proba)
            gain = score - lgbm_score
            print(f"{lgbm_w:4.2f} / {xgb_w:4.2f} => {score:.6f} (Gain: {gain:.6f})")
            
            if score > best_score:
                best_score = score
                best_lgbm_weight = lgbm_w
                best_xgb_weight = xgb_w
                
    print("\n--- EXPERIMENT 4 - COMPLEMENTARITY ANALYSIS ---")
    lgbm_preds = np.argmax(lgbm_val_proba, axis=1)
    xgb_preds = np.argmax(xgb_val_proba, axis=1)
    
    agree = (lgbm_preds == xgb_preds)
    lgbm_correct = (lgbm_preds == y_val)
    xgb_correct = (xgb_preds == y_val)
    
    agree_pct = np.mean(agree) * 100
    both_correct = np.sum(lgbm_correct & xgb_correct)
    lgbm_only_correct = np.sum(lgbm_correct & ~xgb_correct)
    xgb_only_correct = np.sum(xgb_correct & ~lgbm_correct)
    both_wrong = np.sum(~lgbm_correct & ~xgb_correct)
    
    print(f"Prediction agreement %: {agree_pct:.2f}%")
    print(f"Both models correct:    {both_correct}")
    print(f"LGBM correct / XGB wrong: {lgbm_only_correct}")
    print(f"XGB correct / LGBM wrong: {xgb_only_correct}")
    print(f"Both wrong:             {both_wrong}")
    
    print("\n--- FINAL TEST PREDICTION ---")
    if best_score > lgbm_score:
        print(f"ENSEMBLE IMPROVES. Applying weights: LGBM={best_lgbm_weight:.2f}, XGB={best_xgb_weight:.2f}")
        lgbm_test_proba = np.load("lgbm_test_proba.npy")
        xgb_test_proba = np.load("xgb_test_proba.npy")
        
        final_test_proba = (best_lgbm_weight * lgbm_test_proba) + (best_xgb_weight * xgb_test_proba)
        
        np.save("ensemble_val_proba.npy", (best_lgbm_weight * lgbm_val_proba) + (best_xgb_weight * xgb_val_proba))
        np.save("ensemble_test_proba.npy", final_test_proba)
        
        # Save predictions
        test_preds_idx = np.argmax(final_test_proba, axis=1)
        test_preds = [class_order[i] for i in test_preds_idx]
        
        test_df = pd.read_csv("test.csv", usecols=["id"])
        sub = pd.DataFrame({"id": test_df["id"], "target": test_preds})
        sub.to_csv("submission_ensemble.csv", index=False)
        print("Saved ensemble test predictions to submission_ensemble.csv")
    else:
        print("ENSEMBLE DOES NOT IMPROVE - retaining LightGBM as current best model.")

if __name__ == "__main__":
    main()
