import numpy as np
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd

def apply_log_offsets(proba, offsets):
    eps = 1e-15
    log_proba = np.log(proba + eps)
    log_proba += offsets
    exp_proba = np.exp(log_proba - np.max(log_proba, axis=1, keepdims=True))
    return exp_proba / np.sum(exp_proba, axis=1, keepdims=True)

def main():
    val_proba = np.load("lgbm_val_proba.npy")
    y_val = np.load("y_val.npy")
    
    with open("class_order.json", "r") as f:
        class_order = json.load(f)
        
    print(f"Loaded {len(y_val)} samples.")
    
    if y_val.min() == 1:
        print("Adjusting y_val from 1-7 to 0-6")
        y_val_idx = y_val - 1
    else:
        y_val_idx = y_val

    # original score
    preds_orig = np.argmax(val_proba, axis=1)
    f1_orig = f1_score(y_val_idx, preds_orig, average="macro")
    print(f"Original LGBM Macro F1: {f1_orig:.5f}")
    
    offsets = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
    mod_proba = apply_log_offsets(val_proba, offsets)
    preds = np.argmax(mod_proba, axis=1)
    f1_mod = f1_score(y_val_idx, preds, average="macro")
    print(f"Modified LGBM Macro F1: {f1_mod:.5f}")

if __name__ == '__main__':
    main()
