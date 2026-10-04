# -*- coding: utf-8 -*-
"""
lgb_cv.py  -  LightGBM 5-fold stratified CV with OOF/test probability export.

Usage
-----
  python lgb_cv.py                          # defaults
  python lgb_cv.py --alpha 0.25            # softer sample weights
  python lgb_cv.py --no-row-features       # ablate row statistics
  python lgb_cv.py --impute median         # median imputation instead of raw NaN
  python lgb_cv.py --fast                  # higher LR for quick smoke-test
  python lgb_cv.py --run-name my_run       # custom output subdirectory

Ablation runner (separate entry-point):
  python lgb_cv.py --ablation [--fast]
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from subprocess import run as sp_run

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure Unicode prints correctly on Windows terminals (cp1252 -> utf-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LightGBM 5-fold CV – OOF & test probabilities"
    )

    # paths
    p.add_argument("--train", default="train.csv", help="Path to train.csv")
    p.add_argument("--test",  default="test.csv",  help="Path to test.csv")
    p.add_argument("--out-dir", default="artifacts", help="Root output directory")
    p.add_argument(
        "--run-name",
        default=None,
        help="Sub-directory name under --out-dir (auto-generated if omitted)",
    )

    # feature engineering
    p.add_argument(
        "--no-row-features",
        dest="use_row_features",
        action="store_false",
        default=True,
        help="Disable row-level statistics (n_missing, row_mean, …)",
    )
    p.add_argument(
        "--impute",
        choices=["none", "median"],
        default="none",
        help="Imputation strategy; 'none' passes raw NaN to LightGBM",
    )

    # sample weights
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Sample-weight exponent: w_c = (1/freq_c)^alpha. 0 = uniform.",
    )

    # training
    p.add_argument("--folds",      type=int, default=5,  help="Number of CV folds")
    p.add_argument("--num-leaves", type=int, default=32, help="LightGBM num_leaves")
    p.add_argument("--min-child-samples", type=int, default=20)
    p.add_argument("--lr",  type=float, default=0.05,  help="learning_rate")
    p.add_argument("--fast", action="store_true",
                   help="Override lr=0.1 and n_estimators=2000 for quick runs")
    p.add_argument("--n-estimators", type=int, default=5000)
    p.add_argument("--early-stopping-rounds", type=int, default=200)

    # ablation runner mode
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Run all ablation configs and print a comparison table",
    )

    return p


# ──────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering
# ──────────────────────────────────────────────────────────────────────────────
def load_data(train_path: str, test_path: str):
    """Load train and test CSVs and return DataFrames."""
    print(f"Loading {train_path} …")
    df_train = pd.read_csv(train_path)
    print(f"Loading {test_path} …")
    df_test = pd.read_csv(test_path)
    print(f"  train: {df_train.shape}  |  test: {df_test.shape}")
    return df_train, df_test


def get_feature_cols(df_train: pd.DataFrame, df_test: pd.DataFrame) -> list[str]:
    """Return the 174 base feature column names; assert train/test alignment."""
    train_feat = [c for c in df_train.columns if c not in ("id", "target")]
    test_feat  = [c for c in df_test.columns  if c != "id"]

    # Sanity check #3 – same features in same order
    assert train_feat == test_feat, (
        f"Feature column mismatch between train and test!\n"
        f"  train-only: {set(train_feat) - set(test_feat)}\n"
        f"  test-only:  {set(test_feat) - set(train_feat)}"
    )
    return train_feat


def add_row_features(
    X_raw: np.ndarray,
    prefix_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Compute 6 NaN-aware row-level statistics on the RAW 174-feature block.

    Parameters
    ----------
    X_raw      : float array of shape (n, 174) — RAW, before any imputation.
    prefix_cols: list of base feature names (length 174).

    Returns
    -------
    X_aug : array of shape (n, 180)  [original 174 + 6 new cols]
    cols  : updated column name list
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # ignore all-NaN slice
        n_missing = np.isnan(X_raw).sum(axis=1).astype(np.float32)
        row_mean  = np.nanmean(X_raw, axis=1).astype(np.float32)
        row_std   = np.nanstd(X_raw,  axis=1).astype(np.float32)
        row_min   = np.nanmin(X_raw,  axis=1).astype(np.float32)
        row_max   = np.nanmax(X_raw,  axis=1).astype(np.float32)
        row_sum   = np.nansum(X_raw,  axis=1).astype(np.float32)

    stat_block = np.column_stack(
        [n_missing, row_mean, row_std, row_min, row_max, row_sum]
    )
    new_cols = ["n_missing", "row_mean", "row_std", "row_min", "row_max", "row_sum"]

    return np.hstack([X_raw, stat_block]), prefix_cols + new_cols


# ──────────────────────────────────────────────────────────────────────────────
# Sample weights
# ──────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(y_fold: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute per-sample weights from within-fold class frequencies.

      w_c = (1 / freq_c) ^ alpha
      then normalize so mean(weights) == 1.

    alpha=0 → uniform weights (all 1.0).
    """
    if alpha == 0.0:
        return np.ones(len(y_fold), dtype=np.float32)

    classes, counts = np.unique(y_fold, return_counts=True)
    freq = counts / counts.sum()                          # relative frequency
    weight_per_class = (1.0 / freq) ** alpha

    # map each sample to its class weight
    class_to_w = dict(zip(classes, weight_per_class))
    w = np.array([class_to_w[c] for c in y_fold], dtype=np.float64)

    # normalize to mean = 1
    w /= w.mean()
    return w.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def build_lgb_params(args: argparse.Namespace) -> dict:
    """Assemble LightGBM parameter dict from parsed arguments."""
    lr = 0.1 if args.fast else args.lr
    n_est = 100 if args.fast else args.n_estimators

    params = dict(
        objective         = "multiclass",
        num_class         = 7,
        metric            = "multi_logloss",   # early-stopping criterion
        num_leaves        = args.num_leaves,
        min_child_samples = args.min_child_samples,
        learning_rate     = lr,
        feature_fraction  = 0.8,
        bagging_fraction  = 0.8,
        bagging_freq      = 1,
        n_estimators      = n_est,
        n_jobs            = 4,
        random_state      = GLOBAL_SEED,
        verbosity         = -1,
    )
    return params


def train_one_fold(
    fold_idx: int,
    X_train_fold: np.ndarray,
    y_train_fold: np.ndarray,
    X_val_fold:   np.ndarray,
    y_val_fold:   np.ndarray,
    X_test:       np.ndarray,
    sample_weights: np.ndarray,
    lgb_params:   dict,
    early_stop_rounds: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Fit a single LightGBM model, return:
      val_proba  : (n_val, 7)
      test_proba : (n_test, 7)
      best_iter  : int
    """
    t0 = time.time()

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train_fold, y_train_fold,
        sample_weight = sample_weights,
        eval_X        = X_val_fold,          # LightGBM 4.7+ (single eval set as array)
        eval_y        = y_val_fold,
        callbacks     = [
            lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False),
            lgb.log_evaluation(period=0),    # silence per-iteration logs
        ],
    )

    val_proba  = model.predict_proba(X_val_fold)
    test_proba = model.predict_proba(X_test)
    best_iter  = int(model.best_iteration_)
    elapsed    = time.time() - t0

    val_preds  = np.argmax(val_proba, axis=1)
    fold_f1    = f1_score(y_val_fold, val_preds, average="macro")
    print(
        f"  Fold {fold_idx+1} | best_iter={best_iter:4d} | "
        f"val Macro F1={fold_f1:.6f} | {elapsed:.1f}s"
    )

    return val_proba, test_proba, best_iter


# ──────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ──────────────────────────────────────────────────────────────────────────────
def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print row-normalized confusion matrix with class labels 1..7."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    labels = list(range(1, 8))   # class labels 1..7
    header = f"{'':>6}" + "".join(f"  Pred{l}" for l in labels)
    print("\nOOF Confusion Matrix (row-normalised %):")
    print(header)
    for i, row in enumerate(cm_pct):
        row_str = "".join(f"  {v:6.1f}" for v in row)
        print(f"  True{labels[i]}{row_str}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_training(args: argparse.Namespace) -> dict:
    """
    Full 5-fold CV pipeline.  Returns the metrics dict for the ablation runner.
    """
    set_all_seeds(GLOBAL_SEED)

    # ── Output directory ──────────────────────────────────────────────────────
    if args.run_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = "fast_" if args.fast else ""
        rf  = "rf_" if args.use_row_features else "norf_"
        args.run_name = f"lgb_{tag}{rf}a{args.alpha:.2f}_{ts}"

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Run: {args.run_name}")
    print(f"Output dir: {out_dir}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df_train, df_test = load_data(args.train, args.test)
    base_feat_cols    = get_feature_cols(df_train, df_test)

    # Raw feature matrices (dtype float64 for NaN-safe arithmetic)
    X_raw_train = df_train[base_feat_cols].to_numpy(dtype=np.float64)
    X_raw_test  = df_test[base_feat_cols].to_numpy(dtype=np.float64)

    # Labels: remap 1..7 → 0..6
    y_all = df_train["target"].to_numpy(dtype=np.int64) - 1   # 0-indexed

    # ── Row features (computed on raw data BEFORE imputation) ─────────────────
    if args.use_row_features:
        print("Adding 6 row-level features (NaN-aware, on raw 174 cols)…")
        X_aug_train, feat_cols = add_row_features(X_raw_train, base_feat_cols)
        X_aug_test,  _         = add_row_features(X_raw_test,  base_feat_cols)
    else:
        X_aug_train = X_raw_train
        X_aug_test  = X_raw_test
        feat_cols   = base_feat_cols

    print(f"Feature matrix shape: train={X_aug_train.shape}  test={X_aug_test.shape}")
    print(f"  row features: {'ON' if args.use_row_features else 'OFF'}")
    print(f"  imputation  : {args.impute}")
    print(f"  alpha       : {args.alpha}")

    # ── Imputation (optional) ────────────────────────────────────────────────
    # We do NOT fit the imputer globally; it is re-fit per fold to avoid leakage.
    # If impute='none', we just pass the raw arrays.
    use_impute = (args.impute == "median")

    # ── Folds ─────────────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=GLOBAL_SEED)

    # fold_assignments[i] = which fold row i belongs to (0-indexed)
    fold_assignments = np.empty(len(y_all), dtype=np.int32)
    for fold_idx, (_, val_idx) in enumerate(skf.split(X_aug_train, y_all)):
        fold_assignments[val_idx] = fold_idx

    # Save fold assignments (Sanity check: every row in exactly one fold)
    assert np.all(fold_assignments >= 0) and np.all(fold_assignments < args.folds), \
        "Some rows were not assigned to a fold!"
    np.save(out_dir / "folds.npy", fold_assignments)
    print(f"\nFold assignments saved to {out_dir/'folds.npy'}")

    # Sanity check #2 – every train row appears in exactly one validation fold
    val_counts = np.zeros(len(y_all), dtype=np.int32)
    for _, val_idx in skf.split(X_aug_train, y_all):
        val_counts[val_idx] += 1
    assert np.all(val_counts == 1), \
        "Some rows appear in 0 or >1 validation folds!"
    print("✓ Every train row appears in exactly one validation fold.")

    # ── Model params ──────────────────────────────────────────────────────────
    lgb_params = build_lgb_params(args)
    print("\nLightGBM params:")
    for k, v in lgb_params.items():
        print(f"  {k}: {v}")

    # ── Allocate OOF / test probability arrays ────────────────────────────────
    oof_proba   = np.zeros((len(y_all), 7), dtype=np.float64)
    test_probas = np.zeros((len(X_aug_test), 7), dtype=np.float64)   # accumulate

    fold_metrics = []
    best_iterations = []

    # ── CV loop ───────────────────────────────────────────────────────────────
    print(f"\nStarting {args.folds}-fold CV …\n")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_aug_train, y_all)):

        X_tr = X_aug_train[train_idx]
        y_tr = y_all[train_idx]
        X_vl = X_aug_train[val_idx]
        y_vl = y_all[val_idx]
        X_te = X_aug_test  # same test set for each fold

        # Per-fold imputation (fit only on fold's train split)
        if use_impute:
            from sklearn.impute import SimpleImputer
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_vl = imp.transform(X_vl)
            X_te = imp.transform(X_te)

        # Sample weights – computed from fold-train labels only
        sw = compute_sample_weights(y_tr, alpha=args.alpha)

        # Train fold model
        val_proba, test_proba_fold, best_iter = train_one_fold(
            fold_idx   = fold_idx,
            X_train_fold = X_tr,
            y_train_fold = y_tr,
            X_val_fold   = X_vl,
            y_val_fold   = y_vl,
            X_test       = X_te,
            sample_weights  = sw,
            lgb_params   = lgb_params,
            early_stop_rounds = 20 if args.fast else args.early_stopping_rounds,
        )

        # Fill OOF rows
        oof_proba[val_idx] = val_proba

        # Accumulate test probabilities (will average after loop)
        test_probas += test_proba_fold

        # Fold Macro F1
        fold_preds = np.argmax(val_proba, axis=1)
        fold_f1    = f1_score(y_vl, fold_preds, average="macro")
        fold_metrics.append(float(fold_f1))
        best_iterations.append(best_iter)

    # Average test probabilities across folds
    test_probas /= args.folds

    # ── Sanity checks on OOF array ────────────────────────────────────────────
    row_sums = oof_proba.sum(axis=1)
    assert not np.any(np.isnan(oof_proba)), "NaN detected in OOF probabilities!"
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"OOF rows do not sum to 1. Max deviation: {np.abs(row_sums - 1).max():.2e}"
    )
    print("\n✓ OOF sanity checks passed (no NaNs, rows sum to 1 within 1e-4).")

    # ── Overall OOF metrics ───────────────────────────────────────────────────
    oof_preds    = np.argmax(oof_proba, axis=1)
    oof_macro_f1 = float(f1_score(y_all, oof_preds, average="macro"))
    per_class_f1 = f1_score(y_all, oof_preds, average=None).tolist()

    print(f"\n{'='*60}")
    print(f"OOF Macro F1 : {oof_macro_f1:.6f}")
    print(f"Per-fold F1  : {[f'{v:.4f}' for v in fold_metrics]}")
    print(f"Best iters   : {best_iterations}")
    print("\nPer-class OOF F1 (classes 1..7):")
    for cls_label, f1_val in enumerate(per_class_f1, start=1):
        marker = "  ← weakest" if f1_val == min(per_class_f1) else ""
        print(f"  Class {cls_label}: {f1_val:.4f}{marker}")

    print_confusion_matrix(y_all, oof_preds)

    # ── Save artifacts ────────────────────────────────────────────────────────
    np.save(out_dir / "oof_proba.npy",  oof_proba)
    np.save(out_dir / "test_proba.npy", test_probas)
    print(f"Saved oof_proba.npy  {oof_proba.shape}")
    print(f"Saved test_proba.npy {test_probas.shape}")

    metrics = {
        "run_name"        : args.run_name,
        "oof_macro_f1"    : oof_macro_f1,
        "per_fold_f1"     : fold_metrics,
        "per_class_f1"    : {str(i+1): v for i, v in enumerate(per_class_f1)},
        "best_iterations" : best_iterations,
        "n_folds"         : args.folds,
        "n_train"         : int(len(y_all)),
        "n_test"          : int(len(X_aug_test)),
        "n_features"      : int(X_aug_train.shape[1]),
        "use_row_features": args.use_row_features,
        "impute"          : args.impute,
        "alpha"           : args.alpha,
        "fast"            : args.fast,
        "lgb_params"      : lgb_params,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")
    print(f"\nAll artifacts written to: {out_dir}\n")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ──────────────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    # label,            use_row_features, alpha
    ("a) no-rf, a=0",   False,            0.0),
    ("b) rf,    a=0",   True,             0.0),
    ("c) rf,    a=0.5", True,             0.5),
    ("d) rf,    a=0.25",True,             0.25),
    ("e) rf,    a=0.75",True,             0.75),
]


def run_ablation(args: argparse.Namespace) -> None:
    """
    Sequentially run all ablation configs and print a comparison table.
    Each config is run as a subprocess call so log output stays clean.
    """
    script = sys.argv[0]
    results = []

    for label, use_rf, alpha in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"ABLATION CONFIG: {label}")
        print(f"{'='*60}")

        # Build argument list for subprocess
        cmd = [sys.executable, script,
               "--alpha", str(alpha),
               "--train", args.train,
               "--test",  args.test,
               "--out-dir", args.out_dir]

        if not use_rf:
            cmd.append("--no-row-features")
        if args.fast:
            cmd.append("--fast")

        t0 = time.time()
        result = sp_run(cmd, capture_output=False)
        elapsed = time.time() - t0

        # Find the most recent metrics.json written to out_dir
        abl_out = Path(args.out_dir)
        candidates = sorted(
            abl_out.rglob("metrics.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates and result.returncode == 0:
            with open(candidates[0]) as f:
                m = json.load(f)
            results.append({
                "label"       : label,
                "oof_f1"      : m["oof_macro_f1"],
                "class5_f1"   : m["per_class_f1"].get("5", float("nan")),
                "class6_f1"   : m["per_class_f1"].get("6", float("nan")),
                "elapsed_min" : elapsed / 60,
            })
        else:
            results.append({
                "label"       : label,
                "oof_f1"      : float("nan"),
                "class5_f1"   : float("nan"),
                "class6_f1"   : float("nan"),
                "elapsed_min" : elapsed / 60,
            })

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("ABLATION RESULTS")
    print("="*72)
    header = f"{'Config':<22} {'OOF Macro F1':>14} {'Class 5 F1':>12} {'Class 6 F1':>12} {'Time (min)':>11}"
    print(header)
    print("-"*72)
    best_f1 = max(r["oof_f1"] for r in results)
    for r in results:
        marker = " ←" if r["oof_f1"] == best_f1 else ""
        print(
            f"{r['label']:<22} {r['oof_f1']:>14.6f} {r['class5_f1']:>12.4f} "
            f"{r['class6_f1']:>12.4f} {r['elapsed_min']:>10.1f}m{marker}"
        )
    print("="*72)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.ablation:
        run_ablation(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
