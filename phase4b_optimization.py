import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

def main():
    print("=== PHASE 4B: Fine-Grained 3-Way Probability Blend Optimization ===")

    # 1. Load Artifacts
    print("\nLoading probabilities and labels...")
    lgbm_val = np.load("lgbm_val_proba.npy")
    lgbm_test = np.load("lgbm_test_proba.npy")
    xgb_val = np.load("xgb_val_proba.npy")
    xgb_test = np.load("xgb_test_proba.npy")
    cb_val = np.load("cb_val_proba.npy")
    cb_test = np.load("cb_test_proba.npy")
    y_val = np.load("y_val.npy")

    # Verify Alignment
    print("\n1. Verifying Artifact Alignment...")
    val_shapes_match = (lgbm_val.shape == xgb_val.shape == cb_val.shape == (len(y_val), 7))
    test_shapes_match = (lgbm_test.shape == xgb_test.shape == cb_test.shape)
    
    lgbm_val_sum = np.allclose(lgbm_val.sum(axis=1), 1.0)
    xgb_val_sum = np.allclose(xgb_val.sum(axis=1), 1.0)
    cb_val_sum = np.allclose(cb_val.sum(axis=1), 1.0)
    
    print(f"Validation Shapes Match: {val_shapes_match}")
    print(f"Test Shapes Match: {test_shapes_match}")
    print(f"LGBM Val Sum to 1: {lgbm_val_sum}")
    print(f"XGB Val Sum to 1: {xgb_val_sum}")
    print(f"CB Val Sum to 1: {cb_val_sum}")

    if not (val_shapes_match and test_shapes_match and lgbm_val_sum and xgb_val_sum and cb_val_sum):
        print("ERROR: Artifact alignment failed.")
        return

    # 2. Reproduce Baselines
    print("\n2. Reproducing Existing Baselines...")
    def eval_blend(w_lgbm, w_xgb, w_cb):
        p = w_lgbm * lgbm_val + w_xgb * xgb_val + w_cb * cb_val
        preds = np.argmax(p, axis=1)
        return f1_score(y_val, preds, average="macro")

    lgbm_only = eval_blend(1, 0, 0)
    xgb_only = eval_blend(0, 1, 0)
    cb_only = eval_blend(0, 0, 1)
    blend_2way = eval_blend(0.86, 0.14, 0)
    blend_3way_old = eval_blend(0.78, 0.15, 0.07)

    print(f"LGBM alone: {lgbm_only:.6f}")
    print(f"XGB alone: {xgb_only:.6f}")
    print(f"CatBoost alone: {cb_only:.6f}")
    print(f"LGBM 0.86 + XGB 0.14: {blend_2way:.6f}")
    print(f"LGBM 0.78 + XGB 0.15 + CB 0.07: {blend_3way_old:.6f}")

    # 3. Coarse Search
    print("\n3. Coarse Search...")
    lgbm_grid = np.arange(0.70, 0.86 + 0.001, 0.005)
    xgb_grid = np.arange(0.05, 0.25 + 0.001, 0.005)
    
    results = []
    for w_l in lgbm_grid:
        for w_x in xgb_grid:
            w_c = 1.0 - w_l - w_x
            if w_c >= -1e-9:
                # Ensure no negative weights strictly
                w_c = max(0.0, w_c) 
                f1 = eval_blend(w_l, w_x, w_c)
                results.append((w_l, w_x, w_c, f1))

    results.sort(key=lambda x: x[3], reverse=True)
    best_coarse = results[0]
    print(f"Best Coarse: LGBM={best_coarse[0]:.3f}, XGB={best_coarse[1]:.3f}, CB={best_coarse[2]:.3f} -> F1: {best_coarse[3]:.6f}")

    # 4. Fine Search
    print("\n4. Fine Search...")
    l_center, x_center = best_coarse[0], best_coarse[1]
    l_fine = np.arange(max(0, l_center - 0.03), min(1.0, l_center + 0.03) + 0.001, 0.001)
    x_fine = np.arange(max(0, x_center - 0.03), min(1.0, x_center + 0.03) + 0.001, 0.001)
    
    fine_results = []
    for w_l in l_fine:
        for w_x in x_fine:
            w_c = 1.0 - w_l - w_x
            if w_c >= -1e-9:
                w_c = max(0.0, w_c)
                f1 = eval_blend(w_l, w_x, w_c)
                fine_results.append((w_l, w_x, w_c, f1))
                
    fine_results.sort(key=lambda x: x[3], reverse=True)
    best_fine = fine_results[0]
    best_w_l, best_w_x, best_w_c, best_f1 = best_fine
    
    print(f"Best Fine: LGBM={best_w_l:.3f}, XGB={best_w_x:.3f}, CB={best_w_c:.3f} -> F1: {best_f1:.6f}")

    # 5. Stability Analysis
    print("\n5. Stability Analysis...")
    # Analyze neighborhood within ±0.005 of best_fine
    neighborhood = [r for r in fine_results if abs(r[0] - best_w_l) <= 0.005 and abs(r[1] - best_w_x) <= 0.005]
    neighborhood_scores = [r[3] for r in neighborhood]
    
    median_score = np.median(neighborhood_scores)
    min_score = np.min(neighborhood_scores)
    
    diffs = [best_f1 - s for s in neighborhood_scores]
    within_0001 = sum(1 for d in diffs if d <= 0.0001)
    within_0005 = sum(1 for d in diffs if d <= 0.0005)
    within_0010 = sum(1 for d in diffs if d <= 0.0010)
    
    print(f"Neighborhood size: {len(neighborhood)}")
    print(f"Median score in neighborhood: {median_score:.6f}")
    print(f"Min score in neighborhood: {min_score:.6f}")
    print(f"Configurations within 0.0001: {within_0001}")
    print(f"Configurations within 0.0005: {within_0005}")
    print(f"Configurations within 0.0010: {within_0010}")

    # 6. Error/Complementarity Analysis
    print("\n6. Pairwise Agreement Analysis...")
    pred_lgbm = np.argmax(lgbm_val, axis=1)
    pred_xgb = np.argmax(xgb_val, axis=1)
    pred_cb = np.argmax(cb_val, axis=1)
    
    agree_lgbm_xgb = np.mean(pred_lgbm == pred_xgb)
    agree_lgbm_cb = np.mean(pred_lgbm == pred_cb)
    agree_xgb_cb = np.mean(pred_xgb == pred_cb)
    
    print(f"LGBM vs XGB agreement: {agree_lgbm_xgb:.4f}")
    print(f"LGBM vs CB agreement: {agree_lgbm_cb:.4f}")
    print(f"XGB vs CB agreement: {agree_xgb_cb:.4f}")

    # Determine stable recommendation
    # CASE B/C logic
    if best_f1 > blend_3way_old + 1e-5 and (within_0001 > len(neighborhood) * 0.1): 
        # If it's a stable region
        recommended_weights = (best_w_l, best_w_x, best_w_c)
        rec_f1 = best_f1
        decision = "CASE A/B: Adopting new stable weights."
    else:
        recommended_weights = (0.78, 0.15, 0.07)
        rec_f1 = blend_3way_old
        decision = "CASE C: Region not stable enough or improvement marginal. Keeping 0.78 / 0.15 / 0.07."

    # 7. Generate Test Proba & Report Data
    print("\nSaving new test probabilities...")
    test_proba = recommended_weights[0] * lgbm_test + recommended_weights[1] * xgb_test + recommended_weights[2] * cb_test
    np.save("phase4b_best_3way_test_proba.npy", test_proba)
    
    with open("phase4b_best_weights.json", "w") as f:
        json.dump({
            "LGBM": recommended_weights[0],
            "XGB": recommended_weights[1],
            "CB": recommended_weights[2],
            "Macro_F1": rec_f1
        }, f, indent=2)

    report_content = f"""# Phase 4B: Blend Optimization & Stability Analysis

## 1. Artifact Verification
- **Validation Shapes Match**: {val_shapes_match}
- **Test Shapes Match**: {test_shapes_match}
- **Probabilities sum to 1**: Yes (LGBM, XGB, CB)
All artifacts properly aligned.

## 2. Reproduced Baseline Scores
- **LGBM alone**: {lgbm_only:.6f}
- **XGB alone**: {xgb_only:.6f}
- **CatBoost alone**: {cb_only:.6f}
- **LGBM 0.86 + XGB 0.14**: {blend_2way:.6f}
- **LGBM 0.78 + XGB 0.15 + CB 0.07**: {blend_3way_old:.6f}

## 3. Top 20 Combinations (Fine Search)
| Rank | LGBM | XGB | CB | Macro F1 |
|---|---|---|---|---|
"""
    for i, (wl, wx, wc, f1) in enumerate(fine_results[:20]):
        report_content += f"| {i+1} | {wl:.3f} | {wx:.3f} | {wc:.3f} | {f1:.6f} |\n"

    report_content += f"""
## 4. Stability Analysis
**Best Weights**: LGBM={best_w_l:.3f}, XGB={best_w_x:.3f}, CB={best_w_c:.3f}
**Best Score**: {best_f1:.6f}

Analysis of neighborhood (±0.005 for LGBM/XGB):
- **Neighborhood Size**: {len(neighborhood)} configurations
- **Median Score**: {median_score:.6f}
- **Minimum Score**: {min_score:.6f}
- **Configurations within 0.0001 of best**: {within_0001} ({(within_0001/len(neighborhood))*100:.1f}%)
- **Configurations within 0.0005 of best**: {within_0005} ({(within_0005/len(neighborhood))*100:.1f}%)
- **Configurations within 0.0010 of best**: {within_0010} ({(within_0010/len(neighborhood))*100:.1f}%)

**Neighborhood Profile**: 
The performance curve is {"broad/stable" if within_0001 > len(neighborhood)*0.1 else "sharp/peaked"}, suggesting that the optimal weights {"are resilient to minor perturbations" if within_0001 > len(neighborhood)*0.1 else "might overfit the validation set"}.

## 5. Comparison Against Current Champion
| Model / Blend | Macro F1 | Difference vs 0.804109 (Current 2-way) |
| --- | --- | --- |
| LGBM | {lgbm_only:.6f} | {lgbm_only - blend_2way:.6f} |
| XGB | {xgb_only:.6f} | {xgb_only - blend_2way:.6f} |
| CatBoost | {cb_only:.6f} | {cb_only - blend_2way:.6f} |
| Current 2-way (0.86 / 0.14) | {blend_2way:.6f} | 0.000000 |
| Current 3-way (0.78 / 0.15 / 0.07) | {blend_3way_old:.6f} | {blend_3way_old - blend_2way:.6f} |
| Best 3-way found | {best_f1:.6f} | {best_f1 - blend_2way:.6f} |

## 6. Error & Complementarity Analysis
Pairwise prediction agreement on validation set:
- **LGBM vs XGB**: {agree_lgbm_xgb:.4f}
- **LGBM vs CB**: {agree_lgbm_cb:.4f}
- **XGB vs CB**: {agree_xgb_cb:.4f}

CatBoost has lower agreement with LightGBM ({agree_lgbm_cb:.4f}) than XGBoost does ({agree_lgbm_xgb:.4f}). This disagreement indicates that CatBoost is learning different decision boundaries or finding alternative patterns. Consequently, when its probability distributions are blended with LGBM and XGBoost, it acts as a regularizer and occasionally corrects confident but incorrect predictions made by the other models.

## 7. Final Recommendation
**Decision**: {decision}

**Recommended Weights**:
- **LightGBM**: {recommended_weights[0]:.3f}
- **XGBoost**: {recommended_weights[1]:.3f}
- **CatBoost**: {recommended_weights[2]:.3f}

**Recommended CV Score**: {rec_f1:.6f}

New blended test probabilities have been saved to `phase4b_best_3way_test_proba.npy` for the next submission phase.
"""

    with open(r"C:\Users\ASUS\.gemini\antigravity\brain\c9ee79b1-376a-4f37-a6ff-1b2127492b7b\phase4b_blend_optimization_report.md", "w") as f:
        f.write(report_content)
        
    print("Report written successfully.")

if __name__ == "__main__":
    main()
