import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

val_proba = np.load('ensemble_val_proba.npy')
y_val = np.load('y_val.npy')
# y_val is likely 1-7, let's check
if y_val.min() == 1:
    y_val = y_val - 1

def eval_f1(preds):
    return f1_score(y_val, preds, average='macro')

default_preds = np.argmax(val_proba, axis=1)
baseline_f1 = eval_f1(default_preds)
print(f"Baseline Validation Macro F1: {baseline_f1:.5f}")

# Let's try class-specific multipliers, specifically targeting class 6 (index 5)
best_f1 = baseline_f1
best_multiplier = 1.0

for mult in np.linspace(1.0, 1.5, 51):
    mod_proba = val_proba.copy()
    mod_proba[:, 5] *= mult # boost class 6
    preds = np.argmax(mod_proba, axis=1)
    f1 = eval_f1(preds)
    if f1 > best_f1:
        best_f1 = f1
        best_multiplier = mult

print(f"Boosting Class 6: Best Multiplier = {best_multiplier:.3f} -> Validation Macro F1 = {best_f1:.5f}")

# Let's search over all classes
multipliers = np.ones(7)
best_f1 = baseline_f1
import itertools
from scipy.optimize import minimize

def obj_func(weights):
    mod_proba = val_proba * weights
    preds = np.argmax(mod_proba, axis=1)
    return -f1_score(y_val, preds, average='macro')

res = minimize(obj_func, np.ones(7), method='Nelder-Mead', options={'maxiter': 500})
opt_weights = res.x
opt_preds = np.argmax(val_proba * opt_weights, axis=1)
opt_f1 = eval_f1(opt_preds)
print(f"Nelder-Mead Optimization -> Validation Macro F1 = {opt_f1:.5f}")
print(f"Optimal Multipliers: {opt_weights}")

# Phase 6: Selective replacement
# What if we selectively replace our prediction with the 81.162% submission's prediction where we differ and our confidence is low?
# We can simulate this on validation set by assuming "friend prediction" is another model.
# Since we don't have the friend's val prediction, we can only evaluate if we can threshold our confidence.

