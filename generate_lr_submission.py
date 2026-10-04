"""Generate Kaggle submission from the already-validated best Logistic Regression config.

Does not tune hyperparameters, does not read/write P3.5 result files, and does
not use test labels (none expected) for scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42

# Exact best LR trial from phase3_5_results.csv (Macro F1 = 0.795856)
BEST_LR = {
    "imputation": "median",
    "C": 10.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": None,
    "max_iter": 1000,
}
VAL_MACRO_F1 = 0.795856


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")

    id_col = "id"
    target_col = "target"
    train_features = [c for c in train.columns if c not in {id_col, target_col}]
    test_features = [c for c in test.columns if c != id_col]
    feature_match = train_features == test_features

    print("=== STEP 1: TEST / TRAIN INSPECTION ===")
    print(f"train shape: {train.shape}")
    print(f"test shape: {test.shape}")
    print(f"train columns ({len(train.columns)}): {list(train.columns)}")
    print(f"test columns ({len(test.columns)}): {list(test.columns)}")
    print(f"ID column: {id_col}")
    print(f"target column (train only): {target_col}")
    print(f"n train features: {len(train_features)}")
    print(f"n test features: {len(test_features)}")
    print(f"test missing-value count: {int(test.isna().sum().sum())}")
    print(f"train missing-value count: {int(train.isna().sum().sum())}")
    print("test dtypes:")
    print(test.dtypes.value_counts().to_string())
    print("test dtypes (per column):")
    print(test.dtypes.to_string())
    print(f"test feature columns exactly match train feature columns: {feature_match}")
    if not feature_match:
        only_train = sorted(set(train_features) - set(test_features))
        only_test = sorted(set(test_features) - set(train_features))
        print(f"  only in train: {only_train}")
        print(f"  only in test: {only_test}")
        print(f"  order mismatch: {train_features != test_features}")

    valid_classes = sorted(train[target_col].unique().tolist())
    print(f"valid target classes: {valid_classes}")

    sample_path = ROOT / "sample_submission.csv"
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    print(f"sample_submission.csv exists: {sample is not None}")

    print("\n=== STEP 2: BEST LR CONFIG ===")
    print(json.dumps({**BEST_LR, "validation_macro_f1": VAL_MACRO_F1}, indent=2))

    assert feature_match, "Train/test feature columns must match exactly."

    X_train = train[train_features]
    y_train = train[target_col]
    X_test = test[train_features]
    test_ids = test[id_col]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    imputer = SimpleImputer(strategy=BEST_LR["imputation"])
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    print("\n=== STEP 4: TRAIN ON FULL TRAINING DATA ===")
    print(f"training rows: {X_train_sc.shape[0]}")
    print(f"training features after preprocess: {X_train_sc.shape[1]}")
    print(f"test rows after preprocess: {X_test_sc.shape[0]}")

    clf = LogisticRegression(
        C=BEST_LR["C"],
        penalty=BEST_LR["penalty"],
        solver=BEST_LR["solver"],
        class_weight=BEST_LR["class_weight"],
        max_iter=BEST_LR["max_iter"],
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_enc)

    y_pred_enc = clf.predict(X_test_sc)
    y_pred = le.inverse_transform(y_pred_enc)

    print("\n=== STEP 5: PREDICT TEST ===")
    n_pred = len(y_pred)
    n_test = len(test)
    pred_series = pd.Series(y_pred)
    missing_preds = int(pred_series.isna().sum())
    invalid_preds = sorted(set(pred_series.unique()) - set(valid_classes))
    print(f"n predictions: {n_pred}")
    print(f"n test rows: {n_test}")
    print(f"n predictions == n test rows: {n_pred == n_test}")
    print(f"missing predictions: {missing_preds}")
    print(f"invalid predicted classes: {invalid_preds}")
    print(f"all predicted classes valid: {len(invalid_preds) == 0}")
    print("predicted class distribution:")
    dist = pred_series.value_counts().sort_index()
    print(dist.to_string())
    print("predicted class distribution (normalized):")
    print(pred_series.value_counts(normalize=True).sort_index().to_string())

    if sample is not None:
        sub_cols = list(sample.columns)
        submission = sample.copy()
        pred_col = [c for c in sub_cols if c != id_col][0]
        submission[pred_col] = y_pred
        ids_match_sample = submission[id_col].equals(sample[id_col])
        ids_match_test = submission[id_col].equals(test_ids)
    else:
        sub_cols = [id_col, target_col]
        submission = pd.DataFrame({id_col: test_ids, target_col: y_pred})
        ids_match_sample = None
        ids_match_test = submission[id_col].equals(test_ids)

    out_path = ROOT / "submission_lr.csv"
    submission.to_csv(out_path, index=False)

    print("\n=== STEP 7: VALIDATE SUBMISSION ===")
    loaded = pd.read_csv(out_path)
    checks = {
        "correct_n_rows": len(loaded) == n_test,
        "correct_n_columns": loaded.shape[1] == len(sub_cols),
        "correct_column_names": list(loaded.columns) == sub_cols,
        "correct_column_order": list(loaded.columns) == sub_cols,
        "no_missing_values": int(loaded.isna().sum().sum()) == 0,
        "ids_match_test_csv": loaded[id_col].equals(test_ids.reset_index(drop=True)),
        "ids_preserve_original_order": loaded[id_col].tolist() == test_ids.tolist(),
        "prediction_classes_valid": set(loaded.iloc[:, -1].unique()).issubset(set(valid_classes)),
        "n_predictions_equals_n_test": len(loaded) == n_test,
        "no_missing_predictions": int(loaded.iloc[:, -1].isna().sum()) == 0,
        "row_order_preserved": loaded[id_col].tolist() == test_ids.tolist(),
    }
    if ids_match_sample is not None:
        checks["ids_match_sample_submission"] = bool(ids_match_sample)

    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\nFirst 10 rows:")
    print(loaded.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(loaded.tail(10).to_string(index=False))

    dist_lines = "\n".join(
        f"- class {cls}: {int(count)} ({count / n_pred:.4%})"
        for cls, count in dist.items()
    )
    checks_lines = "\n".join(f"- {k}: {v}" for k, v in checks.items())

    report = f"""# Logistic Regression Test Submission Report

True test Macro F1 cannot be calculated because `test.csv` does not contain hidden target labels.
**Do not interpret any figure below as a test F1 score.**

The selected model was already validated on the Phase 3.5 hold-out split
(Validation Macro F1 = {VAL_MACRO_F1:.4f} / 79.59%).

## Data

- Train shape: {train.shape[0]} rows × {train.shape[1]} columns
- Test shape: {test.shape[0]} rows × {test.shape[1]} columns
- ID column: `{id_col}`
- Target column (train only): `{target_col}`
- Number of features: {len(train_features)}
- Feature columns: `{train_features[0]}` … `{train_features[-1]}`
- Test feature columns exactly match train feature columns: {feature_match}
- Test missing-value count: {int(test.isna().sum().sum())}
- Train missing-value count: {int(train.isna().sum().sum())}
- Valid target classes: {valid_classes}

## Best LR configuration (from Phase 3.5 results)

Source: `phase3_5_results.csv` (Logistic Regression trial with Macro F1 = {VAL_MACRO_F1}).

- C: {BEST_LR["C"]}
- solver: {BEST_LR["solver"]}
- penalty: {BEST_LR["penalty"]}
- class_weight: {BEST_LR["class_weight"]}
- max_iter: {BEST_LR["max_iter"]}
- random_state: {SEED}
- n_jobs: -1

## Preprocessing configuration

- Imputation strategy: {BEST_LR["imputation"]} (`SimpleImputer`)
- Scaling strategy: StandardScaler
- Label encoding: `LabelEncoder` fitted on train target only
- Imputer fitted only on full `train.csv` features
- Scaler fitted only on imputed full `train.csv` features
- Test transformed with the same fitted imputer and scaler
- ID column excluded from the model
- Target column excluded from test features
- No additional validation split (model already selected)
- No hyperparameter retuning
- `test.csv` not used for model selection

## Training / prediction counts

- Training row count: {X_train.shape[0]}
- Test row count: {n_test}
- Number of predictions: {n_pred}

## Predicted class distribution

{dist_lines}

## Submission

- File: `{out_path}`
- Submission shape: {loaded.shape[0]} rows × {loaded.shape[1]} columns
- Submission columns: {list(loaded.columns)}
- sample_submission.csv used: {sample is not None}

## Validation checks

{checks_lines}

## First 10 rows

```
{loaded.head(10).to_string(index=False)}
```

## Last 10 rows

```
{loaded.tail(10).to_string(index=False)}
```

## Final status

**submission_lr.csv is ready for Kaggle submission.**
"""
    report_path = ROOT / "lr_test_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {report_path}")
    print("submission_lr.csv is ready for Kaggle submission.")


if __name__ == "__main__":
    main()
