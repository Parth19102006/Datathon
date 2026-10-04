"""
Next-phase optimization pipeline that starts from saved LGBM probability
artifacts and trains XGBoost/CatBoost diversity models.

This intentionally does not overwrite existing best submissions.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_CLASSES = 7
OLD_OFFSETS = np.array([-0.09375, 0.5, -0.25, -0.03125, -0.0625, 0.0, 0.25])
OUT_DIR = Path("next_phase_outputs")


def apply_log_offsets(proba, offsets):
    logp = np.log(proba + 1e-15) + offsets
    exp = np.exp(logp - logp.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def macro_f1(y_true, proba, offsets=None):
    scored = apply_log_offsets(proba, offsets) if offsets is not None else proba
    return f1_score(y_true, np.argmax(scored, axis=1), average="macro")


def sample_weights_balanced(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(classes)
    weight_by_class = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_by_class[c] for c in y], dtype=np.float64)


def optimize_offsets(y_true, proba, start_offsets):
    best_offsets = start_offsets.astype(float).copy()
    best_f1 = macro_f1(y_true, proba, best_offsets)

    history = [{"stage": "start", "f1": float(best_f1), "offsets": best_offsets.tolist()}]
    schedule = [
        ("coarse", np.arange(-0.375, 0.3751, 0.0625), 3),
        ("fine", np.arange(-0.0625, 0.06251, 0.0078125), 3),
    ]

    for stage, deltas, max_passes in schedule:
        for pass_idx in range(max_passes):
            improved = False
            for class_idx in range(N_CLASSES):
                center = best_offsets[class_idx]
                for delta in deltas:
                    trial = best_offsets.copy()
                    trial[class_idx] = center + delta
                    score = macro_f1(y_true, proba, trial)
                    if score > best_f1 + 1e-12:
                        best_f1 = score
                        best_offsets = trial
                        improved = True
            history.append({
                "stage": stage,
                "pass": pass_idx + 1,
                "f1": float(best_f1),
                "offsets": best_offsets.tolist(),
            })
            if not improved:
                break

    return best_offsets, best_f1, history


def write_phase1(y_val, lgbm_val):
    raw_f1 = macro_f1(y_val, lgbm_val)
    offset_f1 = macro_f1(y_val, lgbm_val, OLD_OFFSETS)
    adjusted = apply_log_offsets(lgbm_val, OLD_OFFSETS)
    preds = np.argmax(adjusted, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, preds, labels=np.arange(N_CLASSES), zero_division=0
    )
    cm = confusion_matrix(y_val, preds, labels=np.arange(N_CLASSES))

    mispairs = []
    for true_idx in range(N_CLASSES):
        for pred_idx in range(N_CLASSES):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                mispairs.append({
                    "true": int(true_idx + 1),
                    "pred": int(pred_idx + 1),
                    "count": int(cm[true_idx, pred_idx]),
                })
    mispairs.sort(key=lambda item: item["count"], reverse=True)

    sorted_proba = np.sort(adjusted, axis=1)
    top1 = sorted_proba[:, -1]
    top2 = sorted_proba[:, -2]
    margin = top1 - top2

    phase1 = {
        "lgbm_artifact": "next_lgbm_val_proba.npy",
        "macro_f1_no_offsets": float(raw_f1),
        "macro_f1_old_offsets": float(offset_f1),
        "old_offsets": OLD_OFFSETS.tolist(),
        "per_class": {
            str(i + 1): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(N_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "most_frequent_misclassification_pairs": mispairs[:20],
        "top1_probability": {
            "mean": float(top1.mean()),
            "median": float(np.median(top1)),
            "p10": float(np.quantile(top1, 0.10)),
            "p90": float(np.quantile(top1, 0.90)),
        },
        "top2_probability": {
            "mean": float(top2.mean()),
            "median": float(np.median(top2)),
            "p10": float(np.quantile(top2, 0.10)),
            "p90": float(np.quantile(top2, 0.90)),
        },
        "prediction_margin": {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "p10": float(np.quantile(margin, 0.10)),
            "p90": float(np.quantile(margin, 0.90)),
        },
    }
    (OUT_DIR / "phase1_error_analysis.json").write_text(json.dumps(phase1, indent=2))

    rows = []
    for i in range(N_CLASSES):
        rows.append({
            "class": i + 1,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "phase1_per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=np.arange(1, 8), columns=np.arange(1, 8)).to_csv(
        OUT_DIR / "phase1_confusion_matrix.csv"
    )
    pd.DataFrame(mispairs).to_csv(OUT_DIR / "phase1_misclassification_pairs.csv", index=False)
    np.save(OUT_DIR / "phase1_top1_probability.npy", top1)
    np.save(OUT_DIR / "phase1_top2_probability.npy", top2)
    np.save(OUT_DIR / "phase1_prediction_margin.npy", margin)
    return phase1


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    feature_cols = [c for c in train.columns if c not in ("id", "target")]
    X = train[feature_cols]
    y = train["target"].to_numpy() - 1
    X_test = test[feature_cols]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    saved_y = np.load("y_val.npy")
    if not np.array_equal(y_val, saved_y):
        raise RuntimeError("Validation split does not match y_val.npy.")
    return train, test, X_train, X_val, X_test, y_train, y_val


def train_xgb(X_train_imp, X_val_imp, X_test_imp, y_train, y_val):
    params = {
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 800,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
        "eval_metric": "mlogloss",
    }
    model = XGBClassifier(**params)
    start = time.time()
    model.fit(X_train_imp, y_train, sample_weight=sample_weights_balanced(y_train))
    val = model.predict_proba(X_val_imp)
    test = model.predict_proba(X_test_imp)
    report = {
        "params": params,
        "training_seconds": time.time() - start,
        "standalone_macro_f1": float(macro_f1(y_val, val)),
    }
    np.save("next_xgb_val_proba.npy", val)
    np.save("next_xgb_test_proba.npy", test)
    return val, test, report


def train_catboost(X_train_imp, X_val_imp, X_test_imp, y_train, y_val):
    classes, counts = np.unique(y_train, return_counts=True)
    weights = [len(y_train) / (len(classes) * counts[np.where(classes == c)][0]) for c in classes]
    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "random_seed": RANDOM_STATE,
        "class_weights": weights,
        "verbose": 100,
        "task_type": "CPU",
    }
    model = CatBoostClassifier(**params)
    start = time.time()
    model.fit(X_train_imp, y_train, eval_set=(X_val_imp, y_val), early_stopping_rounds=50)
    val = model.predict_proba(X_val_imp)
    test = model.predict_proba(X_test_imp)
    report = {
        "params": params,
        "best_iteration": int(model.get_best_iteration() or params["iterations"]),
        "training_seconds": time.time() - start,
        "standalone_macro_f1": float(macro_f1(y_val, val)),
    }
    np.save("next_cb_val_proba.npy", val)
    np.save("next_cb_test_proba.npy", test)
    return val, test, report


def blend_search(y_val, models, offsets):
    names = list(models.keys())
    best = {"f1": -1.0}
    rows = []

    if names == ["lgbm", "xgb"]:
        weights_to_try = [(w, 1.0 - w) for w in np.arange(0.50, 0.951, 0.05)]
        weights_to_try += [(w, 1.0 - w) for w in np.arange(0.50, 0.951, 0.01)]
    elif names == ["lgbm", "xgb", "catboost"]:
        weights_to_try = []
        for wl in np.arange(0.50, 0.951, 0.025):
            for wx in np.arange(0.00, 0.401, 0.025):
                wc = 1.0 - wl - wx
                if wc >= -1e-12:
                    weights_to_try.append((wl, wx, max(0.0, wc)))
    else:
        raise ValueError(f"Unsupported blend set: {names}")

    for weights in weights_to_try:
        blend = sum(w * models[name] for w, name in zip(weights, names))
        score = macro_f1(y_val, blend, offsets)
        row = {name: float(w) for name, w in zip(names, weights)}
        row["macro_f1"] = float(score)
        rows.append(row)
        if score > best["f1"]:
            best = {"names": names, "weights": row, "f1": float(score), "proba": blend}

    return best, pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading data and validating split...")
    _, test, X_train, X_val, X_test, y_train, y_val = load_data()

    print("Loading saved LGBM probabilities...")
    lgbm_val = np.load("next_lgbm_val_proba.npy")
    lgbm_test = np.load("next_lgbm_test_proba.npy")
    np.save(OUT_DIR / "lgbm_val_proba_used.npy", lgbm_val)
    np.save(OUT_DIR / "lgbm_test_proba_used.npy", lgbm_test)

    phase1 = write_phase1(y_val, lgbm_val)

    print("Optimizing offsets for LGBM baseline...")
    new_offsets, new_offset_f1, offset_history = optimize_offsets(y_val, lgbm_val, OLD_OFFSETS)
    phase2 = {
        "old_offsets": OLD_OFFSETS.tolist(),
        "new_offsets": new_offsets.tolist(),
        "old_validation_macro_f1": phase1["macro_f1_old_offsets"],
        "new_validation_macro_f1": float(new_offset_f1),
        "improvement": float(new_offset_f1 - phase1["macro_f1_old_offsets"]),
        "history": offset_history,
    }
    (OUT_DIR / "phase2_offset_optimization.json").write_text(json.dumps(phase2, indent=2))
    np.save(OUT_DIR / "phase2_optimized_offsets.npy", new_offsets)

    print("Imputing data for XGB/CatBoost...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    xgb_report_path = OUT_DIR / "phase3_xgb_report.json"
    if Path("next_xgb_val_proba.npy").exists() and Path("next_xgb_test_proba.npy").exists() and xgb_report_path.exists():
        print("Loading cached XGBoost diversity probabilities...")
        xgb_val = np.load("next_xgb_val_proba.npy")
        xgb_test = np.load("next_xgb_test_proba.npy")
        xgb_report = json.loads(xgb_report_path.read_text())
    else:
        print("Training XGBoost diversity model...")
        xgb_val, xgb_test, xgb_report = train_xgb(X_train_imp, X_val_imp, X_test_imp, y_train, y_val)
    lgbm_preds = np.argmax(apply_log_offsets(lgbm_val, new_offsets), axis=1)
    xgb_preds = np.argmax(apply_log_offsets(xgb_val, new_offsets), axis=1)
    xgb_report["agreement_with_lgbm"] = float(np.mean(lgbm_preds == xgb_preds))
    xgb_report["probability_correlation_with_lgbm"] = float(np.corrcoef(lgbm_val.ravel(), xgb_val.ravel())[0, 1])
    lx_best, lx_grid = blend_search(y_val, {"lgbm": lgbm_val, "xgb": xgb_val}, new_offsets)
    lx_grid.to_csv(OUT_DIR / "phase3_lgbm_xgb_blend_grid.csv", index=False)
    xgb_report["best_lgbm_xgb_blend"] = {
        "weights": {k: v for k, v in lx_best["weights"].items() if k != "macro_f1"},
        "macro_f1": lx_best["f1"],
    }
    (OUT_DIR / "phase3_xgb_report.json").write_text(json.dumps(xgb_report, indent=2))

    cb_report_path = OUT_DIR / "phase4_catboost_report.json"
    if Path("next_cb_val_proba.npy").exists() and Path("next_cb_test_proba.npy").exists() and cb_report_path.exists():
        print("Loading cached CatBoost diversity probabilities...")
        cb_val = np.load("next_cb_val_proba.npy")
        cb_test = np.load("next_cb_test_proba.npy")
        cb_report = json.loads(cb_report_path.read_text())
    else:
        print("Training CatBoost diversity model...")
        cb_val, cb_test, cb_report = train_catboost(X_train_imp, X_val_imp, X_test_imp, y_train, y_val)
    cb_preds = np.argmax(apply_log_offsets(cb_val, new_offsets), axis=1)
    cb_report["agreement_with_lgbm"] = float(np.mean(lgbm_preds == cb_preds))
    cb_report["agreement_with_xgb"] = float(np.mean(xgb_preds == cb_preds))
    cb_report["probability_correlation_with_lgbm"] = float(np.corrcoef(lgbm_val.ravel(), cb_val.ravel())[0, 1])
    cb_report["probability_correlation_with_xgb"] = float(np.corrcoef(xgb_val.ravel(), cb_val.ravel())[0, 1])
    (OUT_DIR / "phase4_catboost_report.json").write_text(json.dumps(cb_report, indent=2))

    print("Searching final blends and per-blend offsets...")
    lxc_best, lxc_grid = blend_search(
        y_val, {"lgbm": lgbm_val, "xgb": xgb_val, "catboost": cb_val}, new_offsets
    )
    lxc_grid.to_csv(OUT_DIR / "phase5_lgbm_xgb_catboost_blend_grid.csv", index=False)

    final_rows = [{
        "experiment": "Current LGBM baseline",
        "local_macro_f1": float(phase1["macro_f1_old_offsets"]),
        "improvement": 0.0,
        "offsets": OLD_OFFSETS.tolist(),
    }]

    candidates = [
        ("Optimized offsets", lgbm_val, lgbm_test, new_offsets),
        ("LGBM + XGB", lx_best["proba"], (
            lx_best["weights"]["lgbm"] * lgbm_test + lx_best["weights"]["xgb"] * xgb_test
        ), new_offsets),
        ("LGBM + XGB + CatBoost", lxc_best["proba"], (
            lxc_best["weights"]["lgbm"] * lgbm_test
            + lxc_best["weights"]["xgb"] * xgb_test
            + lxc_best["weights"]["catboost"] * cb_test
        ), new_offsets),
    ]

    final_candidates = []
    baseline = phase1["macro_f1_old_offsets"]
    for name, val_proba, test_proba, start_offsets in candidates:
        opt_offsets, opt_f1, _ = optimize_offsets(y_val, val_proba, start_offsets)
        final_rows.append({
            "experiment": name,
            "local_macro_f1": float(opt_f1),
            "improvement": float(opt_f1 - baseline),
            "offsets": opt_offsets.tolist(),
        })
        final_candidates.append((opt_f1, name, val_proba, test_proba, opt_offsets))

    final_rows.append({
        "experiment": "XGB standalone",
        "local_macro_f1": xgb_report["standalone_macro_f1"],
        "improvement": xgb_report["standalone_macro_f1"] - baseline,
        "offsets": None,
    })
    final_rows.append({
        "experiment": "CatBoost standalone",
        "local_macro_f1": cb_report["standalone_macro_f1"],
        "improvement": cb_report["standalone_macro_f1"] - baseline,
        "offsets": None,
    })

    best_f1, best_name, _, best_test_proba, best_offsets = max(final_candidates, key=lambda item: item[0])
    final_pred = np.argmax(apply_log_offsets(best_test_proba, best_offsets), axis=1) + 1
    submission = pd.DataFrame({"id": test["id"], "target": final_pred})
    submission.to_csv("submission_next.csv", index=False)

    final_config = {
        "best_experiment": best_name,
        "best_local_macro_f1": float(best_f1),
        "baseline_local_macro_f1": float(baseline),
        "improvement": float(best_f1 - baseline),
        "offsets": best_offsets.tolist(),
        "lgbm_artifacts": ["next_lgbm_val_proba.npy", "next_lgbm_test_proba.npy"],
        "xgb_artifacts": ["next_xgb_val_proba.npy", "next_xgb_test_proba.npy"],
        "catboost_artifacts": ["next_cb_val_proba.npy", "next_cb_test_proba.npy"],
        "submission": "submission_next.csv",
        "note": (
            "LightGBM retraining was not performed in this script; it starts from saved "
            "LGBM probability artifacts to avoid modifying existing best submissions."
        ),
    }
    (OUT_DIR / "phase5_final_results.json").write_text(json.dumps({
        "experiments": final_rows,
        "best_config": final_config,
    }, indent=2))
    pd.DataFrame(final_rows).drop(columns=["offsets"]).to_csv(
        OUT_DIR / "phase5_experiment_table.csv", index=False
    )
    Path("next_best_config.json").write_text(json.dumps(final_config, indent=2))

    print(json.dumps(final_config, indent=2))


if __name__ == "__main__":
    main()
