"""
all_phases.py – Complete 5-phase optimization pipeline.
Loads data ONCE, trains LGBM/XGB/CB, optimizes offsets, searches blends,
and generates submission_next.csv.

DO NOT modify existing best submission files.
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix)
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def apply_log_offsets(proba, offsets):
    """Add offsets in log-space and softmax back."""
    eps = 1e-15
    logp = np.log(proba + eps) + offsets
    expp = np.exp(logp - logp.max(axis=1, keepdims=True))
    return expp / expp.sum(axis=1, keepdims=True)

def macro_f1(y_true, proba):
    return f1_score(y_true, np.argmax(proba, axis=1), average="macro")

def macro_f1_with_offsets(y_true, proba, offsets):
    return macro_f1(y_true, apply_log_offsets(proba, offsets))

def get_sqrt_balanced_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    w_map = {c: np.sqrt(n / (k * cnt)) for c, cnt in zip(classes, counts)}
    return np.array([w_map[c] for c in y], dtype=np.float64)

def get_balanced_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    w_map = {c: n / (k * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w_map[c] for c in y], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════════════
# Data Loading (done ONCE)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING DATA (one-time)")
print("=" * 70)
t_start = time.time()

df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")
print(f"  train {df_train.shape}, test {df_test.shape}  ({time.time()-t_start:.1f}s)")

feat_cols = [c for c in df_train.columns if c not in ("id", "target")]
X_all     = df_train[feat_cols].values
y_all     = df_train["target"].values - 1          # 0-indexed (0…6)
X_test_raw = df_test[feat_cols].values

# Exact same split
X_tr, X_vl, y_tr, y_vl = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

# Verify alignment with saved y_val
saved_y = np.load("y_val.npy")
assert np.array_equal(y_vl, saved_y), "y_val mismatch with saved y_val.npy!"
print("  [OK] y_val matches saved y_val.npy")

imp = SimpleImputer(strategy="median")
X_tr_imp  = imp.fit_transform(X_tr)
X_vl_imp  = imp.transform(X_vl)
X_te_imp  = imp.transform(X_test_raw)
print(f"  Imputation done.  Total load time: {time.time()-t_start:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Reproduce best LGBM ensemble & error analysis
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1 — Train best LGBM ensemble & validation error analysis")
print("=" * 70)

base_params = dict(
    objective="multiclass", num_class=7,
    learning_rate=0.055, num_leaves=31, max_bin=191,
    min_child_samples=30, colsample_bytree=0.9,
    subsample=0.85, subsample_freq=1,
    reg_lambda=2.0, reg_alpha=0.05,
    n_jobs=4, random_state=42, verbose=-1,
)

# Model 1 – balanced class weights, 997 trees
print("  Training Model 1 (balanced, 997 trees)...")
t0 = time.time()
p1 = {**base_params, "n_estimators": 997, "class_weight": "balanced"}
m1 = lgb.LGBMClassifier(**p1)
m1.fit(X_tr_imp, y_tr)
m1_val = m1.predict_proba(X_vl_imp)
m1_test = m1.predict_proba(X_te_imp)
print(f"    done ({time.time()-t0:.1f}s)")

# Model 2 – sqrt-balanced sample weights, 995 trees
print("  Training Model 2 (sqrt-balanced, 995 trees)...")
t0 = time.time()
p2 = {**base_params, "n_estimators": 995}
m2 = lgb.LGBMClassifier(**p2)
sw_sqrt = get_sqrt_balanced_weights(y_tr)
m2.fit(X_tr_imp, y_tr, sample_weight=sw_sqrt)
m2_val = m2.predict_proba(X_vl_imp)
m2_test = m2.predict_proba(X_te_imp)
print(f"    done ({time.time()-t0:.1f}s)")

# 50:50 blend
lgbm_val  = 0.5 * m1_val  + 0.5 * m2_val
lgbm_test = 0.5 * m1_test + 0.5 * m2_test

np.save("next_lgbm_val_proba.npy",  lgbm_val)
np.save("next_lgbm_test_proba.npy", lgbm_test)

offsets_old = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
f1_raw  = macro_f1(y_vl, lgbm_val)
f1_off  = macro_f1_with_offsets(y_vl, lgbm_val, offsets_old)
print(f"\n  LGBM ensemble (no offsets):  {f1_raw:.6f}")
print(f"  LGBM ensemble (old offsets): {f1_off:.6f}")

# Error analysis
mod_proba = apply_log_offsets(lgbm_val, offsets_old)
preds = np.argmax(mod_proba, axis=1)
prec, rec, f1_pc, sup = precision_recall_fscore_support(
    y_vl, preds, labels=np.arange(7), zero_division=0
)
cm = confusion_matrix(y_vl, preds, labels=np.arange(7))

top1_prob = np.max(mod_proba, axis=1)
sorted_proba = np.sort(mod_proba, axis=1)
top2_prob = sorted_proba[:, -2]
margin = top1_prob - top2_prob

# Misclassification pairs
mispairs = []
for i in range(7):
    for j in range(7):
        if i != j and cm[i, j] > 0:
            mispairs.append(((i+1, j+1), int(cm[i, j])))
mispairs.sort(key=lambda x: x[1], reverse=True)

print(f"\n  Per-class metrics:")
print(f"  {'Class':>5} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
for i in range(7):
    print(f"  {i+1:>5} {prec[i]:>7.4f} {rec[i]:>7.4f} {f1_pc[i]:>7.4f} {sup[i]:>8}")

print(f"\n  Top-5 misclassification pairs (true→pred, count):")
for (ti, pi), cnt in mispairs[:10]:
    print(f"    Class {ti} → Class {pi}: {cnt}")

print(f"\n  Avg top-1 prob: {np.mean(top1_prob):.4f}")
print(f"  Avg top-2 prob: {np.mean(top2_prob):.4f}")
print(f"  Avg margin:     {np.mean(margin):.4f}")

phase1_report = {
    "macro_f1_no_offsets": float(f1_raw),
    "macro_f1_with_offsets": float(f1_off),
    "per_class": {str(i+1): {"precision": float(prec[i]), "recall": float(rec[i]),
                              "f1": float(f1_pc[i]), "support": int(sup[i])} for i in range(7)},
    "confusion_matrix": cm.tolist(),
    "top_misclass_pairs": [{"true": t, "pred": p, "count": c} for (t,p),c in mispairs[:15]],
    "avg_top1_prob": float(np.mean(top1_prob)),
    "avg_top2_prob": float(np.mean(top2_prob)),
    "avg_margin": float(np.mean(margin)),
}
with open("next_phase1_report.json", "w") as f:
    json.dump(phase1_report, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Optimize offsets
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2 — Optimize class offsets (coarse→fine coordinate descent)")
print("=" * 70)

best_offsets = offsets_old.copy()
best_f1_off = f1_off

# Coarse pass
for iteration in range(3):
    improved = False
    for ci in range(7):
        cur = best_offsets[ci]
        for delta in np.arange(-0.5, 0.51, 0.0625):
            trial = best_offsets.copy()
            trial[ci] = cur + delta
            sc = macro_f1_with_offsets(y_vl, lgbm_val, trial)
            if sc > best_f1_off + 1e-8:
                best_f1_off = sc
                best_offsets = trial.copy()
                improved = True
    if not improved:
        break

print(f"  After coarse: {best_f1_off:.6f}")

# Fine pass
for iteration in range(3):
    improved = False
    for ci in range(7):
        cur = best_offsets[ci]
        for delta in np.arange(-0.0625, 0.0626, 0.0078125):
            trial = best_offsets.copy()
            trial[ci] = cur + delta
            sc = macro_f1_with_offsets(y_vl, lgbm_val, trial)
            if sc > best_f1_off + 1e-8:
                best_f1_off = sc
                best_offsets = trial.copy()
                improved = True
    if not improved:
        break

print(f"  After fine:   {best_f1_off:.6f}")
print(f"  Old offsets:  {offsets_old.tolist()}")
print(f"  New offsets:  {best_offsets.tolist()}")
print(f"  Old F1:       {f1_off:.6f}")
print(f"  New F1:       {best_f1_off:.6f}")
print(f"  Improvement:  {best_f1_off - f1_off:.6f}")

phase2_report = {
    "old_offsets": offsets_old.tolist(),
    "new_offsets": best_offsets.tolist(),
    "old_f1": float(f1_off),
    "new_f1": float(best_f1_off),
    "improvement": float(best_f1_off - f1_off),
}
with open("next_phase2_report.json", "w") as f:
    json.dump(phase2_report, f, indent=2)

optimized_offsets = best_offsets.copy()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: XGBoost diversity model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3 — XGBoost diversity model")
print("=" * 70)

sw_balanced = get_balanced_weights(y_tr)

xgb_configs = [
    {"max_depth": 6, "min_child_weight": 3, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8, "n_estimators": 1000,
     "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 7, "min_child_weight": 5, "learning_rate": 0.04,
     "subsample": 0.85, "colsample_bytree": 0.85, "n_estimators": 1200,
     "reg_alpha": 0.05, "reg_lambda": 2.0},
]

best_xgb_f1 = 0
best_xgb_val = None
best_xgb_test = None

for ci, cfg in enumerate(xgb_configs):
    print(f"  Config {ci+1}: {cfg}")
    t0 = time.time()
    xm = XGBClassifier(
        objective="multi:softprob", num_class=7,
        tree_method="hist", random_state=42, n_jobs=4,
        use_label_encoder=False, eval_metric="mlogloss",
        **cfg
    )
    xm.fit(X_tr_imp, y_tr, sample_weight=sw_balanced)
    xgb_v = xm.predict_proba(X_vl_imp)
    xgb_t = xm.predict_proba(X_te_imp)
    sc = macro_f1(y_vl, xgb_v)
    print(f"    XGB standalone F1: {sc:.6f}  ({time.time()-t0:.1f}s)")
    if sc > best_xgb_f1:
        best_xgb_f1 = sc
        best_xgb_val = xgb_v
        best_xgb_test = xgb_t

np.save("next_xgb_val_proba.npy",  best_xgb_val)
np.save("next_xgb_test_proba.npy", best_xgb_test)

# Agreement & blending
xgb_preds = np.argmax(best_xgb_val, axis=1)
lgbm_preds = np.argmax(lgbm_val, axis=1)
agree = np.mean(xgb_preds == lgbm_preds)
print(f"\n  LGBM/XGB prediction agreement: {agree:.4f}")

print(f"\n  Blend search (LGBM+XGB):")
blend_results = []
for wl in np.arange(0.5, 0.96, 0.05):
    wx = 1.0 - wl
    p = wl * lgbm_val + wx * best_xgb_val
    sc = macro_f1(y_vl, p)
    blend_results.append((wl, wx, sc))
    print(f"    LGBM {wl:.2f} / XGB {wx:.2f} → {sc:.6f}")

# Fine search around the best
blend_results.sort(key=lambda x: x[2], reverse=True)
best_wl = blend_results[0][0]
for wl in np.arange(max(0.5, best_wl - 0.05), min(0.95, best_wl + 0.05) + 0.001, 0.01):
    wx = 1.0 - wl
    p = wl * lgbm_val + wx * best_xgb_val
    sc = macro_f1(y_vl, p)
    blend_results.append((wl, wx, sc))

blend_results.sort(key=lambda x: x[2], reverse=True)
best_lgbm_xgb = blend_results[0]
print(f"\n  Best LGBM/XGB blend: {best_lgbm_xgb[0]:.2f}/{best_lgbm_xgb[1]:.2f} → {best_lgbm_xgb[2]:.6f}")

phase3_report = {
    "xgb_standalone_f1": float(best_xgb_f1),
    "agreement_with_lgbm": float(agree),
    "best_lgbm_xgb_blend": {"lgbm": float(best_lgbm_xgb[0]),
                             "xgb": float(best_lgbm_xgb[1]),
                             "f1": float(best_lgbm_xgb[2])},
}
with open("next_phase3_report.json", "w") as f:
    json.dump(phase3_report, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: CatBoost diversity test
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4 — CatBoost diversity test")
print("=" * 70)

classes, counts = np.unique(y_tr, return_counts=True)
cb_class_w = {int(c): float(len(y_tr) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

t0 = time.time()
cb = CatBoostClassifier(
    iterations=1000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3, loss_function="MultiClass",
    eval_metric="TotalF1:average=Macro",
    random_seed=42, class_weights=cb_class_w,
    verbose=100, task_type="CPU",
    early_stopping_rounds=50,
)
cb.fit(X_tr_imp, y_tr, eval_set=(X_vl_imp, y_vl))
cb_val  = cb.predict_proba(X_vl_imp)
cb_test = cb.predict_proba(X_te_imp)
cb_f1_standalone = macro_f1(y_vl, cb_val)
print(f"  CatBoost standalone F1: {cb_f1_standalone:.6f}  ({time.time()-t0:.1f}s)")

np.save("next_cb_val_proba.npy",  cb_val)
np.save("next_cb_test_proba.npy", cb_test)

cb_preds = np.argmax(cb_val, axis=1)
agree_lgbm_cb = np.mean(lgbm_preds == cb_preds)
agree_xgb_cb  = np.mean(xgb_preds == cb_preds)
print(f"  LGBM/CB agreement: {agree_lgbm_cb:.4f}")
print(f"  XGB/CB  agreement: {agree_xgb_cb:.4f}")

phase4_report = {
    "cb_standalone_f1": float(cb_f1_standalone),
    "lgbm_cb_agreement": float(agree_lgbm_cb),
    "xgb_cb_agreement": float(agree_xgb_cb),
}
with open("next_phase4_report.json", "w") as f:
    json.dump(phase4_report, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Final ensemble search
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5 — Final ensemble search (weights + offsets)")
print("=" * 70)

# We also have m1_val, m2_val as separate LGBM sub-models
candidate_models = {
    "lgbm_blend": lgbm_val,
    "lgbm_bal":   m1_val,
    "lgbm_sqrt":  m2_val,
    "xgb":        best_xgb_val,
    "catboost":   cb_val,
}
candidate_test = {
    "lgbm_blend": lgbm_test,
    "lgbm_bal":   m1_test,
    "lgbm_sqrt":  m2_test,
    "xgb":        best_xgb_test,
    "catboost":   cb_test,
}

# ── Search 1: 3-model blend (lgbm_blend, xgb, catboost) ──
print("\n  Search 1: LGBM + XGB + CatBoost")
best_3w = {"f1": 0}
for wl in np.arange(0.50, 0.96, 0.02):
    for wx in np.arange(0.02, 0.50, 0.02):
        wc = 1.0 - wl - wx
        if wc < -1e-9:
            continue
        wc = max(0, wc)
        p = wl * lgbm_val + wx * best_xgb_val + wc * cb_val
        sc = macro_f1(y_vl, p)
        if sc > best_3w["f1"]:
            best_3w = {"wl": wl, "wx": wx, "wc": wc, "f1": sc}

# Fine-tune
center = best_3w
for wl in np.arange(max(0.4, center["wl"]-0.03), min(0.99, center["wl"]+0.03)+0.001, 0.005):
    for wx in np.arange(max(0.01, center["wx"]-0.03), min(0.50, center["wx"]+0.03)+0.001, 0.005):
        wc = 1.0 - wl - wx
        if wc < -1e-9:
            continue
        wc = max(0, wc)
        p = wl * lgbm_val + wx * best_xgb_val + wc * cb_val
        sc = macro_f1(y_vl, p)
        if sc > best_3w["f1"]:
            best_3w = {"wl": wl, "wx": wx, "wc": wc, "f1": sc}

print(f"    Best 3-way: LGBM {best_3w['wl']:.3f} / XGB {best_3w['wx']:.3f} / CB {best_3w['wc']:.3f}")
print(f"    F1 (no offsets): {best_3w['f1']:.6f}")

# ── Search 2: 4-model blend (lgbm_bal, lgbm_sqrt, xgb, catboost) ──
print("\n  Search 2: LGBM_bal + LGBM_sqrt + XGB + CatBoost")
best_4w = {"f1": 0}
for w1 in np.arange(0.20, 0.60, 0.05):
    for w2 in np.arange(0.20, 0.60, 0.05):
        for w3 in np.arange(0.02, 0.30, 0.04):
            w4 = 1.0 - w1 - w2 - w3
            if w4 < -1e-9 or w4 > 0.50:
                continue
            w4 = max(0, w4)
            p = w1*m1_val + w2*m2_val + w3*best_xgb_val + w4*cb_val
            sc = macro_f1(y_vl, p)
            if sc > best_4w["f1"]:
                best_4w = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "f1": sc}

# Fine-tune 4-way
if best_4w["f1"] > 0:
    c4 = best_4w
    for w1 in np.arange(max(0.1, c4["w1"]-0.05), min(0.7, c4["w1"]+0.05)+0.001, 0.01):
        for w2 in np.arange(max(0.1, c4["w2"]-0.05), min(0.7, c4["w2"]+0.05)+0.001, 0.01):
            for w3 in np.arange(max(0.01, c4["w3"]-0.03), min(0.3, c4["w3"]+0.03)+0.001, 0.01):
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-9 or w4 > 0.50:
                    continue
                w4 = max(0, w4)
                p = w1*m1_val + w2*m2_val + w3*best_xgb_val + w4*cb_val
                sc = macro_f1(y_vl, p)
                if sc > best_4w["f1"]:
                    best_4w = {"w1": w1, "w2": w2, "w3": w3, "w4": w4, "f1": sc}

    print(f"    Best 4-way: bal {best_4w['w1']:.3f} / sqrt {best_4w['w2']:.3f} / "
          f"xgb {best_4w['w3']:.3f} / cb {best_4w['w4']:.3f}")
    print(f"    F1 (no offsets): {best_4w['f1']:.6f}")

# ── Apply offsets to best configurations ──
print("\n  Applying optimized offsets to each candidate:")

experiments = []

# 1. LGBM only + old offsets
f1_lgbm_old = macro_f1_with_offsets(y_vl, lgbm_val, offsets_old)
experiments.append(("LGBM baseline (old offsets)", f1_lgbm_old, None, None))

# 2. LGBM only + new offsets
f1_lgbm_new = macro_f1_with_offsets(y_vl, lgbm_val, optimized_offsets)
experiments.append(("LGBM + optimized offsets", f1_lgbm_new, None, None))

# 3. Best LGBM/XGB blend + optimized offsets
bx = best_lgbm_xgb
p_lx = bx[0] * lgbm_val + bx[1] * best_xgb_val
f1_lx = macro_f1_with_offsets(y_vl, p_lx, optimized_offsets)
experiments.append(("LGBM+XGB + opt offsets", f1_lx,
                    {"lgbm": bx[0], "xgb": bx[1]},
                    lambda: bx[0]*lgbm_test + bx[1]*best_xgb_test))

# 4. 3-way blend + optimized offsets
p_3 = best_3w["wl"]*lgbm_val + best_3w["wx"]*best_xgb_val + best_3w["wc"]*cb_val
f1_3 = macro_f1_with_offsets(y_vl, p_3, optimized_offsets)
experiments.append(("LGBM+XGB+CB + opt offsets", f1_3,
                    {"lgbm": best_3w["wl"], "xgb": best_3w["wx"], "cb": best_3w["wc"]},
                    lambda: best_3w["wl"]*lgbm_test + best_3w["wx"]*best_xgb_test + best_3w["wc"]*cb_test))

# 5. 4-way blend + optimized offsets
if best_4w["f1"] > 0:
    p_4 = best_4w["w1"]*m1_val + best_4w["w2"]*m2_val + best_4w["w3"]*best_xgb_val + best_4w["w4"]*cb_val
    f1_4 = macro_f1_with_offsets(y_vl, p_4, optimized_offsets)
    experiments.append(("4-model + opt offsets", f1_4,
                        {"lgbm_bal": best_4w["w1"], "lgbm_sqrt": best_4w["w2"],
                         "xgb": best_4w["w3"], "cb": best_4w["w4"]},
                        lambda: best_4w["w1"]*m1_test + best_4w["w2"]*m2_test +
                                best_4w["w3"]*best_xgb_test + best_4w["w4"]*cb_test))

# Also re-optimize offsets for the best raw blend
# Find best raw blend first
raw_blends = [
    ("lgbm_only", lgbm_val, lgbm_test),
    ("lgbm_xgb", bx[0]*lgbm_val + bx[1]*best_xgb_val,
                  bx[0]*lgbm_test + bx[1]*best_xgb_test),
    ("3way", best_3w["wl"]*lgbm_val + best_3w["wx"]*best_xgb_val + best_3w["wc"]*cb_val,
             best_3w["wl"]*lgbm_test + best_3w["wx"]*best_xgb_test + best_3w["wc"]*cb_test),
]
if best_4w["f1"] > 0:
    raw_blends.append(("4way",
        best_4w["w1"]*m1_val + best_4w["w2"]*m2_val + best_4w["w3"]*best_xgb_val + best_4w["w4"]*cb_val,
        best_4w["w1"]*m1_test + best_4w["w2"]*m2_test + best_4w["w3"]*best_xgb_test + best_4w["w4"]*cb_test))

# For each raw blend, re-optimize offsets from scratch
print("\n  Re-optimizing offsets for each blend candidate:")
final_candidates = []
for name, vp, tp in raw_blends:
    off = optimized_offsets.copy()
    bf1 = macro_f1_with_offsets(y_vl, vp, off)
    # Coordinate descent again
    for _ in range(3):
        imp_flag = False
        for ci in range(7):
            cur = off[ci]
            for delta in np.arange(-0.5, 0.51, 0.03125):
                trial = off.copy()
                trial[ci] = cur + delta
                sc = macro_f1_with_offsets(y_vl, vp, trial)
                if sc > bf1 + 1e-8:
                    bf1 = sc
                    off = trial.copy()
                    imp_flag = True
            # fine
            cur = off[ci]
            for delta in np.arange(-0.03125, 0.03126, 0.00390625):
                trial = off.copy()
                trial[ci] = cur + delta
                sc = macro_f1_with_offsets(y_vl, vp, trial)
                if sc > bf1 + 1e-8:
                    bf1 = sc
                    off = trial.copy()
                    imp_flag = True
        if not imp_flag:
            break
    print(f"    {name:15s}: F1 = {bf1:.6f}  offsets = {off.tolist()}")
    final_candidates.append((name, bf1, off, vp, tp))

# Print experiment table
print("\n" + "=" * 70)
print("EXPERIMENT TABLE")
print("=" * 70)
baseline_f1_val = f1_lgbm_old
print(f"{'Experiment':<35} {'Local Macro-F1':>15} {'Improvement':>12}")
print("-" * 65)
print(f"{'LGBM baseline (old offsets)':<35} {baseline_f1_val:>15.6f} {'—':>12}")
for name, sc, _ , _ , _ in final_candidates:
    diff = sc - baseline_f1_val
    print(f"{name:<35} {sc:>15.6f} {diff:>+12.6f}")
print(f"{'XGB standalone':<35} {best_xgb_f1:>15.6f} {best_xgb_f1-baseline_f1_val:>+12.6f}")
print(f"{'CatBoost standalone':<35} {cb_f1_standalone:>15.6f} {cb_f1_standalone-baseline_f1_val:>+12.6f}")

# ── Select best and generate submission ──
final_candidates.sort(key=lambda x: x[1], reverse=True)
best = final_candidates[0]
best_name, best_f1_final, best_off, best_vp, best_tp = best

print(f"\n  ★ BEST: {best_name} → {best_f1_final:.6f}")
print(f"    Offsets: {best_off.tolist()}")

# Generate submission
final_test_proba = apply_log_offsets(best_tp, best_off)
final_preds_idx = np.argmax(final_test_proba, axis=1)
# Map back to 1-7
final_preds = final_preds_idx + 1

sub = pd.DataFrame({"id": df_test["id"], "target": final_preds})
assert len(sub) == len(df_test)
assert sub["target"].isna().sum() == 0
assert set(sub["target"].unique()).issubset(set(range(1, 8)))

sub.to_csv("submission_next.csv", index=False)
print(f"\n  Saved submission_next.csv ({len(sub)} rows)")
print(f"  Prediction distribution:")
for v, c in sorted(sub["target"].value_counts().items()):
    print(f"    Class {v}: {c} ({c/len(sub)*100:.1f}%)")

# Save best config
best_config = {
    "best_name": best_name,
    "best_f1": float(best_f1_final),
    "offsets": best_off.tolist(),
    "baseline_f1": float(baseline_f1_val),
    "improvement": float(best_f1_final - baseline_f1_val),
}
with open("next_best_config.json", "w") as f:
    json.dump(best_config, f, indent=2)

print(f"\n{'='*70}")
print(f"DONE. Best local Macro-F1: {best_f1_final:.6f}")
print(f"Improvement over baseline: {best_f1_final - baseline_f1_val:+.6f}")
print(f"{'='*70}")
