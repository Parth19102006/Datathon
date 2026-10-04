import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

def main():
    print("=== PHASE 4C: Fast Compute-Efficient LightGBM Optimization ===")
    
    # 1. Load Data & Reproduce Validation Split
    print("\n--- Step 1: Loading Data & Exact Split ---")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    
    id_col = "id"
    target_col = "target"
    
    feature_cols = [c for c in df_train.columns if c not in [id_col, target_col]]
    test_feature_cols = [c for c in df_test.columns if c != id_col]
    
    X = df_train[feature_cols]
    y = df_train[target_col]
    X_test = df_test[test_feature_cols]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Preprocessing: Median Imputation
    print("Applying Median Imputer...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    # Verify y_val matches existing y_val.npy
    saved_y_val = np.load("y_val.npy")
    assert np.array_equal(y_val, saved_y_val), "Validation target mismatch!"
    
    # 2. Establish & Reproduce Baseline
    print("\n--- Step 2: Reproducing Baseline LightGBM ---")
    with open("phase3_5_best_models.json", "r") as f:
        best_models = json.load(f)
    base_params = best_models["LightGBM"]["best_params"].copy()
    
    t0 = time.time()
    base_lgbm = LGBMClassifier(**base_params)
    base_lgbm.fit(X_train_imp, y_train)
    base_time = time.time() - t0
    
    base_val_proba = base_lgbm.predict_proba(X_val_imp)
    base_preds = np.argmax(base_val_proba, axis=1)
    base_macro_f1 = f1_score(y_val, base_preds, average="macro")
    print(f"Baseline LGBM Val Macro F1: {base_macro_f1:.6f} (Fit Time: {base_time:.2f}s)")
    
    # 3. Fast Targeted Search
    print("\n--- Step 3: Targeted Small Search ---")
    
    target_configs = [
        {"num_leaves": 32, "min_child_samples": 20, "learning_rate": 0.075},
        {"num_leaves": 64, "min_child_samples": 20, "learning_rate": 0.075},
        {"num_leaves": 96, "min_child_samples": 20, "learning_rate": 0.05},
        {"num_leaves": 128, "min_child_samples": 20, "learning_rate": 0.03},
        
        {"num_leaves": 64, "min_child_samples": 50, "learning_rate": 0.075},
        {"num_leaves": 96, "min_child_samples": 50, "learning_rate": 0.05},
        {"num_leaves": 128, "min_child_samples": 50, "learning_rate": 0.03},
        
        {"num_leaves": 64, "min_child_samples": 100, "learning_rate": 0.075},
        {"num_leaves": 96, "min_child_samples": 100, "learning_rate": 0.05},
        {"num_leaves": 128, "min_child_samples": 100, "learning_rate": 0.03},
    ]
    
    experiment_records = []
    best_f1 = base_macro_f1
    best_params_found = None
    best_val_proba = None
    best_model = None
    
    for i, cfg in enumerate(target_configs):
        p = base_params.copy()
        p.update(cfg)
        p["n_estimators"] = 800  # Cap to prevent running forever
        
        t_start = time.time()
        model = LGBMClassifier(**p)
        callbacks = [early_stopping(30, verbose=False), log_evaluation(0)]
        
        model.fit(X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], callbacks=callbacks)
        
        fit_duration = time.time() - t_start
        val_proba = model.predict_proba(X_val_imp)
        val_preds = np.argmax(val_proba, axis=1)
        macro_f1 = f1_score(y_val, val_preds, average="macro")
        
        print(f"[Config {i+1}] leaves={cfg['num_leaves']}, min_child={cfg['min_child_samples']}, lr={cfg['learning_rate']} -> F1: {macro_f1:.6f} | Time: {fit_duration:.1f}s | Iter: {model.best_iteration_}")
        
        experiment_records.append({
            "config_id": f"CFG_{i+1}",
            "params": cfg,
            "macro_f1": float(macro_f1),
            "time": fit_duration,
            "best_iter": model.best_iteration_
        })
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_params_found = p
            best_val_proba = val_proba
            best_model = model

    print(f"\nSearch Complete. Best F1 Found: {best_f1:.6f}")
    
    improvement = best_f1 - base_macro_f1
    print(f"Improvement over baseline: {improvement:.6f}")
    
    if improvement < 0.0005:
        print("\nDecision: Improvement is < 0.0005. Stopping LightGBM optimization.")
        report = f"# Phase 4C: Fast LightGBM Optimization (ABORTED)\n\n"
        report += f"**Baseline LGBM F1**: {base_macro_f1:.6f}\n"
        report += f"**Best Config F1**: {best_f1:.6f}\n"
        report += f"**Improvement**: +{improvement:.6f}\n\n"
        report += "Improvement was less than 0.0005 threshold. Optimization aborted to prevent overfitting. Keeping existing ensemble.\n"
        with open("phase4c_fast_optimization_report.md", "w") as f:
            f.write(report)
        return
        
    print("\nMeaningful improvement found. Generating Test Probabilities...")
    opt_lgbm_test_proba = best_model.predict_proba(X_test_imp)
    
    np.save("phase4c_best_lgbm_val_proba.npy", best_val_proba)
    np.save("phase4c_best_lgbm_test_proba.npy", opt_lgbm_test_proba)
    
    with open("phase4c_best_lgbm_config.json", "w") as f:
        json.dump({"best_params": best_params_found, "macro_f1": best_f1}, f, indent=2)

    # 4. Ensemble Re-optimization
    print("\n--- Step 4: Local Focused Ensemble Re-optimization ---")
    xgb_val = np.load("xgb_val_proba.npy")
    cb_val = np.load("cb_val_proba.npy")
    
    # Focused search around LGBM 0.785, XGB 0.145, CB 0.070
    l_fine = np.arange(0.74, 0.83 + 0.0005, 0.005)
    x_fine = np.arange(0.10, 0.20 + 0.0005, 0.005)
    
    best_blend_f1 = 0
    best_weights = (0,0,0)
    
    for w_l in l_fine:
        for w_x in x_fine:
            w_c = 1.0 - w_l - w_x
            if w_c >= 0.0:
                p = w_l * best_val_proba + w_x * xgb_val + w_c * cb_val
                f1 = f1_score(y_val, np.argmax(p, axis=1), average="macro")
                if f1 > best_blend_f1:
                    best_blend_f1 = f1
                    best_weights = (w_l, w_x, w_c)
                    
    print(f"Best New 3-way Blend: LGBM={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CB={best_weights[2]:.3f} -> Macro F1: {best_blend_f1:.6f}")
    
    prev_champion_f1 = 0.804643
    ens_improvement = best_blend_f1 - prev_champion_f1
    print(f"Ensemble Improvement vs 0.804643: {ens_improvement:.6f}")
    
    if ens_improvement > 0.0001:
        print("Decision: Meaningful ensemble improvement. Saving new champion blend.")
        xgb_test = np.load("xgb_test_proba.npy")
        cb_test = np.load("cb_test_proba.npy")
        new_blend_test_proba = best_weights[0] * opt_lgbm_test_proba + best_weights[1] * xgb_test + best_weights[2] * cb_test
        np.save("phase4c_fast_best_3way_test_proba.npy", new_blend_test_proba)
    else:
        print("Decision: No meaningful ensemble improvement. Keeping old ensemble.")
        
    report = f"""# Phase 4C: Fast LightGBM Optimization Report

## Baseline
- **Baseline LGBM F1**: {base_macro_f1:.6f}

## LightGBM Optimization Results
"""
    for rec in experiment_records:
        report += f"- Config: {rec['params']} -> **F1: {rec['macro_f1']:.6f}** (Iter: {rec['best_iter']}, Time: {rec['time']:.1f}s)\n"
        
    report += f"""
## Best LightGBM Model
- **Best Params**: {json.dumps(best_params_found)}
- **Optimized LGBM F1**: {best_f1:.6f}
- **Improvement over Baseline LGBM**: {improvement:.6f}

## Ensemble Re-optimization
- **Previous Champion (0.785 / 0.145 / 0.070)**: 0.804643
- **New Best 3-way Weights**: LGBM={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CB={best_weights[2]:.3f}
- **New Ensemble F1**: {best_blend_f1:.6f}
- **Ensemble Improvement**: {ens_improvement:.6f}

## Conclusion & Question Answered
**Q:** Did optimizing LightGBM produce a robust improvement to the final ensemble, and is it large enough to justify replacing the current 0.804643 champion?

**A:** {"YES" if ens_improvement > 0.0001 else "NO"}. The LightGBM optimization {"succeeded in boosting" if ens_improvement > 0.0001 else "failed to boost"} the ensemble score meaningfully. {"New ensemble saved." if ens_improvement > 0.0001 else "We retain the existing 0.804643 ensemble."}
"""

    with open("phase4c_fast_optimization_report.md", "w") as f:
        f.write(report)
    print("\nReport saved to phase4c_fast_optimization_report.md")

if __name__ == "__main__":
    main()
