import numpy as np
from sklearn.metrics import f1_score
import json

def apply_log_offsets(proba, offsets):
    eps = 1e-15
    log_proba = np.log(proba + eps)
    log_proba += offsets
    exp_proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
    return exp_proba / np.sum(exp_proba, axis=1, keepdims=True)

def eval_offsets(proba, y_true, offsets):
    mod_proba = apply_log_offsets(proba, offsets)
    preds = np.argmax(mod_proba, axis=1)
    return f1_score(y_true, preds, average="macro")

def main():
    print("Loading Phase 1 probabilities...")
    val_proba = np.load("val_proba_lgbm_best.npy")
    y_val = np.load("y_val_exact.npy")
    
    initial_offsets = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
    baseline_f1 = eval_offsets(val_proba, y_val, initial_offsets)
    
    print(f"Old offsets: {initial_offsets.tolist()}")
    print(f"Old validation Macro-F1: {baseline_f1:.6f}")
    
    best_offsets = initial_offsets.copy()
    best_f1 = baseline_f1
    
    # Coarse coordinate descent
    print("\nStarting Coarse Coordinate Descent...")
    improved = True
    iteration = 0
    while improved and iteration < 5:
        improved = False
        iteration += 1
        
        for i in range(7):
            best_val_for_class = best_offsets[i]
            
            # Coarse search around the current best offset for class i
            for step in np.arange(-0.4, 0.41, 0.1):
                test_val = best_offsets[i] + step
                test_offsets = best_offsets.copy()
                test_offsets[i] = test_val
                
                f1 = eval_offsets(val_proba, y_val, test_offsets)
                if f1 > best_f1:
                    best_f1 = f1
                    best_val_for_class = test_val
                    improved = True
                    
            best_offsets[i] = best_val_for_class
            
    print(f"After Coarse Search F1: {best_f1:.6f}")
    print(f"Coarse Offsets: {best_offsets.tolist()}")
    
    # Fine coordinate descent
    print("\nStarting Fine Coordinate Descent...")
    improved = True
    iteration = 0
    while improved and iteration < 5:
        improved = False
        iteration += 1
        
        for i in range(7):
            best_val_for_class = best_offsets[i]
            
            # Fine search around the current best offset for class i
            for step in np.arange(-0.05, 0.051, 0.0125):
                test_val = best_offsets[i] + step
                test_offsets = best_offsets.copy()
                test_offsets[i] = test_val
                
                f1 = eval_offsets(val_proba, y_val, test_offsets)
                if f1 > best_f1:
                    best_f1 = f1
                    best_val_for_class = test_val
                    improved = True
                    
            best_offsets[i] = best_val_for_class
            
    print(f"\nFinal New offsets: {best_offsets.tolist()}")
    print(f"New validation Macro-F1: {best_f1:.6f}")
    print(f"Improvement: {best_f1 - baseline_f1:.6f}")
    
    np.save("optimized_offsets.npy", best_offsets)
    with open("phase2_results.json", "w") as f:
        json.dump({
            "old_offsets": initial_offsets.tolist(),
            "new_offsets": best_offsets.tolist(),
            "old_f1": baseline_f1,
            "new_f1": best_f1,
            "improvement": best_f1 - baseline_f1
        }, f, indent=2)

if __name__ == "__main__":
    main()
