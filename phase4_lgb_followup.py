"""One follow-up: LightGBM with class_weight=balanced and more trees.

Same P3/P3.5 split. Appends to phase4_experiments.csv. Does not touch test.csv
for selection. Does not overwrite phase3_5 files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
SEED = 42
LOG = ROOT / "phase4_experiments.csv"
FEATURE_COLS = [f"f{i}" for i in range(1, 175)]


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    X = train[FEATURE_COLS]
    y_enc = LabelEncoder().fit_transform(train["target"])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )
    params = dict(
        objective="multiclass",
        num_class=7,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=96,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
        class_weight="balanced",
    )
    clf = lgb.LGBMClassifier(**params)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    t_fit = time.time() - t0
    pred = clf.predict(X_va)
    elapsed = time.time() - t0 + t_fit  # includes predict in second interval incorrectly
    t_pred_start = time.time()
    # pred already computed; recompute time as fit + predict
    elapsed = time.time() - t0
    # redo timing cleanly was lost; report wall from fit start including predict
    row = dict(
        experiment="LGB_BALANCED_400",
        model="LightGBM",
        preprocessing="none (native NaN handling); ID/target excluded",
        features="all 174",
        hyperparameters=", ".join(f"{k}={v}" for k, v in params.items() if k != "verbosity"),
        macro_f1=float(f1_score(y_va, pred, average="macro")),
        weighted_f1=float(f1_score(y_va, pred, average="weighted")),
        accuracy=float(accuracy_score(y_va, pred)),
        time_sec=round(t_fit + (time.time() - t_pred_start), 3),
        notes="Follow-up: balanced + more trees after LGB_BASE ignored rare classes",
    )
    # fix time: t_fit is fit only; predict was before t_pred_start. Use t_fit as lower bound.
    row["time_sec"] = round(t_fit, 3)
    print(
        f"LGB_BALANCED_400  MF1={row['macro_f1']:.4f}  WF1={row['weighted_f1']:.4f}  "
        f"Acc={row['accuracy']:.4f}  fit={t_fit:.1f}s"
    )
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=False, index=False)

    results = pd.read_csv(LOG).sort_values("macro_f1", ascending=False)
    best = results.iloc[0]
    payload = {
        "best_experiment": best["experiment"],
        "best_model": best["model"],
        "best_macro_f1": float(best["macro_f1"]),
        "row": best.to_dict(),
        "class_labels": [1, 2, 3, 4, 5, 6, 7],
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "competition_metric": "macro_f1 (primary); also tracked weighted_f1 and accuracy",
    }
    (ROOT / "phase4_best_config.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(results[["experiment", "model", "macro_f1", "weighted_f1", "accuracy", "time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
