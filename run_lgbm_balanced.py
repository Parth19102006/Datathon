import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

import common


def build_parser():
    parser = argparse.ArgumentParser(description="Honest 5-fold LGBM balanced CV.")
    parser.add_argument("--run-name", default="LGBM_balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    return parser


def build_params(seed: int, fast: bool):
    params = {
        "objective": "multiclass",
        "num_class": common.N_CLASSES,
        "learning_rate": 0.07921743455371703,
        "n_estimators": 500,
        "num_leaves": 30,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.774,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": seed,
        "verbose": -1,
    }
    if fast:
        params["learning_rate"] *= 2.0
        params["n_estimators"] = 80
    return params


def main():
    args = build_parser().parse_args()
    common.set_seed(args.seed)

    start = time.time()
    out_dir = Path("artifacts") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = common.load_train_test(args.train, args.test)
    feature_cols = common.get_feature_columns(train, test)
    X, y, X_test = common.load_arrays(train, test, feature_cols)
    folds = common.load_folds(y)

    assert X.shape == (228039, len(feature_cols)), "Unexpected train feature shape."
    assert X_test.shape == (97731, len(feature_cols)), "Unexpected test feature shape."
    assert common.ID_COL not in feature_cols, "id leaked into model features."

    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError(
            "LightGBM could not be imported. Stage 0 cannot proceed until the "
            "local LightGBM installation is usable."
        ) from exc

    params = build_params(args.seed, args.fast)
    oof_proba = np.zeros((len(y), common.N_CLASSES), dtype=np.float64)
    test_proba = np.zeros((len(test), common.N_CLASSES), dtype=np.float64)
    oof_seen = np.zeros(len(y), dtype=bool)
    fold_scores = []

    for fold_id in range(5):
        fold_start = time.time()
        train_idx, val_idx = common.split_indices(folds, fold_id)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr_imp, y_tr)

        val_proba = model.predict_proba(X_val_imp)
        fold_test_proba = model.predict_proba(X_test_imp)
        common.assert_probability_array(val_proba, (len(val_idx), common.N_CLASSES), f"fold_{fold_id}_val_proba")
        common.assert_probability_array(fold_test_proba, (len(test), common.N_CLASSES), f"fold_{fold_id}_test_proba")

        oof_proba[val_idx] = val_proba
        oof_seen[val_idx] = True
        test_proba += fold_test_proba / 5.0
        fold_f1 = common.macro_f1(y_val, val_proba)
        fold_scores.append(fold_f1)
        print(f"fold={fold_id} macro_f1={fold_f1:.6f} seconds={time.time() - fold_start:.1f}")

    common.assert_oof_original_order(oof_seen)
    common.assert_probability_array(oof_proba, (228039, common.N_CLASSES), "oof_proba")
    common.assert_probability_array(test_proba, (97731, common.N_CLASSES), "test_proba")

    oof_macro_f1 = common.macro_f1(y, oof_proba)
    metrics = {
        "run_name": args.run_name,
        "model": "LGBM_balanced",
        "seed": args.seed,
        "fast": args.fast,
        "pooled_oof_macro_f1": oof_macro_f1,
        "per_class_oof_f1": common.per_class_f1(y, oof_proba),
        "per_fold_macro_f1": fold_scores,
        "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "std_fold_macro_f1": float(np.std(fold_scores)),
        "used_iteration_count": params["n_estimators"],
        "model_params": params,
        "runtime_seconds": time.time() - start,
        "n_train": int(len(y)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "confusion_matrix": common.confusion_matrix(y, oof_proba),
    }

    np.save(out_dir / "oof_proba.npy", oof_proba)
    np.save(out_dir / "test_proba.npy", test_proba)
    common.save_json(out_dir / "metrics.json", metrics)
    print(f"pooled_oof_macro_f1={oof_macro_f1:.6f}")
    print(f"runtime_seconds={metrics['runtime_seconds']:.1f}")
    print(f"artifacts={out_dir}")


if __name__ == "__main__":
    main()
