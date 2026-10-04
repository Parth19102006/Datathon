"""
Phase 3.5 — Hyperparameter Tuning  (v3)
- LR: DONE (results already saved, skip)
- HGB: class_weight fixed to None (balanced caused 3+ hour trials), max_iter capped at 200
- LightGBM: Optuna 50 trials
Appends to existing phase3_5_results.csv
"""

import json
import time
import warnings
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phase3_5_run_v3.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

SEED = 42
N_TRIALS_HGB = 30   # reduced — HGB is slow
N_TRIALS_LGB = 50

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD ALREADY-SAVED LR RESULTS
# ─────────────────────────────────────────────────────────────────────────────
existing = pd.read_csv("phase3_5_results.csv")
all_results = existing.to_dict("records")
log.info("Loaded %d existing results from phase3_5_results.csv", len(all_results))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOAD & SPLIT  (identical to P3 / previous run)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Loading data ...")
df = pd.read_csv("train.csv")
X = df.drop(columns=["id", "target"])
y = df["target"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

log.info("Splitting data (stratified 80/20, seed=42) ...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
)
log.info("Train: %s  |  Val: %s", X_train.shape, X_val.shape)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
best_per_family = {}

def record(model_name, params, y_true, y_pred, t_train, t_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    wf1 = f1_score(y_true, y_pred, average="weighted")
    all_results.append(dict(
        Model=model_name,
        Params=json.dumps(params, default=str),
        Macro_F1=round(mf1, 6),
        Weighted_F1=round(wf1, 6),
        Accuracy=round(acc, 6),
        Training_Time=round(t_train, 3),
        Prediction_Time=round(t_pred, 4),
    ))
    return mf1, acc, wf1

def save_checkpoint():
    pd.DataFrame(all_results).to_csv("phase3_5_results.csv", index=False)

# Pull best LR from existing results
lr_rows = [r for r in all_results if r["Model"] == "Logistic Regression"]
if lr_rows:
    best_lr = max(lr_rows, key=lambda r: r["Macro_F1"])
    log.info("Best existing LR: Macro F1=%.4f  params=%s", best_lr["Macro_F1"], best_lr["Params"])
    best_per_family["Logistic Regression"] = {
        "params": json.loads(best_lr["Params"]),
        "macro_f1": best_lr["Macro_F1"],
        "y_pred": None,   # we'll re-predict at report time
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4.  HISTGRADIENTBOOSTING — Optuna (no class_weight="balanced", capped iter)
# ─────────────────────────────────────────────────────────────────────────────
log.info("="*60)
log.info("HISTGRADIENTBOOSTING — Optuna (%d trials, class_weight=None, max_iter<=200)", N_TRIALS_HGB)
log.info("="*60)

best_hgb_f1  = -1
best_hgb_row = None

def hgb_objective(trial):
    global best_hgb_f1, best_hgb_row
    params = dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        max_iter          = trial.suggest_int("max_iter", 100, 200, step=50),
        max_leaf_nodes    = trial.suggest_int("max_leaf_nodes", 20, 60),
        max_depth         = trial.suggest_int("max_depth", 4, 8),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 20, 100),
        l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
        class_weight      = None,   # "balanced" causes 3-hour trials on this dataset
    )
    clf = HistGradientBoostingClassifier(**params, random_state=SEED)
    t0 = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_val)
    t_pred = time.time() - t0

    mf1, acc, wf1 = record("HistGradientBoosting", params, y_val, y_pred, t_train, t_pred)
    log.info("  HGB #%d -> MF1=%.4f  lr=%.4f  leaves=%d  depth=%d  iter=%d  [%.1fs]",
             trial.number, mf1, params["learning_rate"],
             params["max_leaf_nodes"], params["max_depth"], params["max_iter"], t_train)

    if mf1 > best_hgb_f1:
        best_hgb_f1  = mf1
        best_hgb_row = dict(params=params, macro_f1=mf1, y_pred=y_pred.tolist())
        save_checkpoint()
    return mf1

hgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
hgb_study.optimize(hgb_objective, n_trials=N_TRIALS_HGB, show_progress_bar=False)
log.info("Best HGB Macro F1 = %.4f", best_hgb_f1)
best_per_family["HistGradientBoosting"] = best_hgb_row
save_checkpoint()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LIGHTGBM — Optuna
# ─────────────────────────────────────────────────────────────────────────────
log.info("="*60)
log.info("LIGHTGBM — Optuna (%d trials)", N_TRIALS_LGB)
log.info("="*60)

imp_lgb = SimpleImputer(strategy="median")
X_train_lgb = imp_lgb.fit_transform(X_train)
X_val_lgb   = imp_lgb.transform(X_val)

best_lgb_f1  = -1
best_lgb_row = None

def lgb_objective(trial):
    global best_lgb_f1, best_lgb_row
    params = dict(
        objective         = "multiclass",
        num_class         = 7,
        metric            = "multi_logloss",
        verbosity         = -1,
        n_jobs            = -1,
        random_state      = SEED,
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        n_estimators      = trial.suggest_int("n_estimators", 100, 500, step=100),
        num_leaves        = trial.suggest_int("num_leaves", 20, 200),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        class_weight      = trial.suggest_categorical("class_weight", ["balanced", None]),
    )
    clf = lgb.LGBMClassifier(**params)
    t0 = time.time()
    clf.fit(X_train_lgb, y_train)
    t_train = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_val_lgb)
    t_pred = time.time() - t0

    mf1, acc, wf1 = record("LightGBM", params, y_val, y_pred, t_train, t_pred)
    log.info("  LGB #%d -> MF1=%.4f  lr=%.4f  leaves=%d  n_est=%d  cw=%s  [%.1fs]",
             trial.number, mf1, params["learning_rate"],
             params["num_leaves"], params["n_estimators"],
             params["class_weight"], t_train)

    if mf1 > best_lgb_f1:
        best_lgb_f1  = mf1
        best_lgb_row = dict(params=params, macro_f1=mf1, y_pred=y_pred.tolist())
        save_checkpoint()
    return mf1

lgb_study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
lgb_study.optimize(lgb_objective, n_trials=N_TRIALS_LGB, show_progress_bar=False)
log.info("Best LGB Macro F1 = %.4f", best_lgb_f1)
best_per_family["LightGBM"] = best_lgb_row
save_checkpoint()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RE-PREDICT BEST LR FOR REPORT (using saved best params)
# ─────────────────────────────────────────────────────────────────────────────
log.info("Re-predicting best LR for per-class report ...")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

best_lr_params = best_per_family["Logistic Regression"]["params"]
imp_type = best_lr_params["imputation"]
imp = SimpleImputer(strategy=imp_type)
X_train_imp = imp.fit_transform(X_train)
X_val_imp   = imp.transform(X_val)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imp)
X_val_sc   = sc.transform(X_val_imp)

lr_best = LogisticRegression(
    C=best_lr_params["C"],
    penalty=best_lr_params["penalty"],
    solver="lbfgs",
    class_weight=best_lr_params["class_weight"],
    max_iter=1000,
    random_state=SEED,
    n_jobs=-1,
)
lr_best.fit(X_train_sc, y_train)
lr_y_pred = lr_best.predict(X_val_sc)
best_per_family["Logistic Regression"]["y_pred"] = lr_y_pred.tolist()
log.info("LR re-prediction done.")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  SAVE FINAL OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
log.info("Saving final outputs ...")
class_labels = [str(c) for c in le.classes_]

family_reports = {}
family_cms     = {}
for fam, row in best_per_family.items():
    if row is None:
        continue
    yp = np.array(row["y_pred"])
    family_reports[fam] = classification_report(
        y_val, yp, target_names=class_labels, output_dict=True
    )
    family_cms[fam] = confusion_matrix(y_val, yp).tolist()

with open("phase3_5_reports.json", "w") as f:
    json.dump({"classes": class_labels,
               "classification_reports": family_reports,
               "confusion_matrices": family_cms}, f, indent=2)

best_models_out = {
    fam: {"best_params": row["params"], "best_macro_f1": row["macro_f1"]}
    for fam, row in best_per_family.items() if row
}
with open("phase3_5_best_models.json", "w") as f:
    json.dump(best_models_out, f, indent=2, default=str)

results_df = (
    pd.DataFrame(all_results)
    .sort_values("Macro_F1", ascending=False)
    .reset_index(drop=True)
)
results_df.to_csv("phase3_5_results.csv", index=False)
log.info("All outputs saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 10 CONFIGURATIONS BY MACRO F1")
print("="*70)
print(results_df.head(10)[["Model","Macro_F1","Weighted_F1","Accuracy","Training_Time"]].to_markdown(index=True))

print("\n" + "="*70)
print("BEST CONFIG PER FAMILY")
print("="*70)
for fam, info in best_models_out.items():
    print(f"\n  {fam}  |  Macro F1: {info['best_macro_f1']:.4f}")
    for k, v in info["best_params"].items():
        print(f"    {k:<25}: {v}")

print("\n" + "="*70)
print("PER-CLASS REPORT — BEST OF EACH FAMILY")
print("="*70)
for fam, report in family_reports.items():
    print(f"\n--- {fam} ---")
    hdr = f"  {'Class':<10}{'Precision':>11}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print(hdr)
    for cls in class_labels:
        r = report[cls]
        print(f"  {cls:<10}{r['precision']:>11.4f}{r['recall']:>10.4f}"
              f"{r['f1-score']:>10.4f}{r['support']:>10.0f}")
    r = report["macro avg"]
    print(f"  {'macro avg':<10}{r['precision']:>11.4f}{r['recall']:>10.4f}{r['f1-score']:>10.4f}")

print("\n" + "="*70)
print("CONFUSION MATRICES — BEST PER FAMILY")
print("="*70)
for fam, cm in family_cms.items():
    print(f"\n--- {fam} ---  (rows=actual  cols=predicted  labels={class_labels})")
    print(pd.DataFrame(cm, index=class_labels, columns=class_labels).to_string())

print("\nPhase 3.5 complete.")
