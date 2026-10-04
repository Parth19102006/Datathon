import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def main():
    print("=== STEP 1 & 2: Load Data and Reproduce Validation Split ===")
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    id_col = "id"
    target_col = "target"

    feature_cols = [c for c in df_train.columns if c not in [id_col, target_col]]
    test_feature_cols = [c for c in df_test.columns if c != id_col]

    assert feature_cols == test_feature_cols, "Train and Test feature columns do not match!"

    X = df_train[feature_cols]
    y = df_train[target_col]
    X_test = df_test[test_feature_cols]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_order = le.classes_.tolist()

    # Stratified 80/20 split with fixed random_state=42
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape:   {X_val.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Classes:     {class_order}")

    # Save y_val.npy and class_order.json
    np.save("y_val.npy", y_val)
    with open("class_order.json", "w") as f:
        json.dump(class_order, f, indent=2)
    print("Saved y_val.npy and class_order.json")

    # Preprocessing for LightGBM (Median Imputer)
    print("\nPreparing Imputed Data for LightGBM...")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    # === STEP 3 & 4: LightGBM Probabilities & Verification ===
    print("\n=== STEP 3 & 4: LightGBM Probabilities & Verification ===")
    with open("phase3_5_best_models.json", "r") as f:
        best_models = json.load(f)
    lgbm_params = best_models["LightGBM"]["best_params"]
    print("LightGBM Final Params:", lgbm_params)

    lgbm_model = LGBMClassifier(**lgbm_params)
    t0 = time.time()
    lgbm_model.fit(X_train_imp, y_train)
    print(f"LightGBM fitted in {time.time() - t0:.2f}s")

    lgbm_val_proba = lgbm_model.predict_proba(X_val_imp)
    lgbm_test_proba = lgbm_model.predict_proba(X_test_imp)

    np.save("lgbm_val_proba.npy", lgbm_val_proba)
    np.save("lgbm_test_proba.npy", lgbm_test_proba)
    print("Saved lgbm_val_proba.npy and lgbm_test_proba.npy")

    lgbm_val_preds = np.argmax(lgbm_val_proba, axis=1)
    lgbm_macro_f1 = f1_score(y_val, lgbm_val_preds, average="macro")
    lgbm_acc = accuracy_score(y_val, lgbm_val_preds)
    print(f"LightGBM Val Macro F1: {lgbm_macro_f1:.6f} (Expected ~0.802625)")
    print(f"LightGBM Val Accuracy: {lgbm_acc:.6f}")

    # === STEP 5: XGBoost Probabilities & Verification ===
    print("\n=== STEP 5: XGBoost Probabilities & Verification ===")
    # Baseline XGBoost configuration from Phase 3
    xgb_params = {
        "n_estimators": 100,
        "n_jobs": -1,
        "random_state": 42,
        "objective": "multi:softprob",
        "num_class": 7,
        "eval_metric": "mlogloss"
    }
    print("XGBoost Final Params:", xgb_params)

    xgb_model = XGBClassifier(**xgb_params)
    t0 = time.time()
    xgb_model.fit(X_train, y_train)
    print(f"XGBoost fitted in {time.time() - t0:.2f}s")

    xgb_val_proba = xgb_model.predict_proba(X_val)
    xgb_test_proba = xgb_model.predict_proba(X_test)

    np.save("xgb_val_proba.npy", xgb_val_proba)
    np.save("xgb_test_proba.npy", xgb_test_proba)
    print("Saved xgb_val_proba.npy and xgb_test_proba.npy")

    xgb_val_preds = np.argmax(xgb_val_proba, axis=1)
    xgb_macro_f1 = f1_score(y_val, xgb_val_preds, average="macro")
    xgb_acc = accuracy_score(y_val, xgb_val_preds)
    print(f"XGBoost Val Macro F1: {xgb_macro_f1:.6f} (Expected ~0.755838)")
    print(f"XGBoost Val Accuracy: {xgb_acc:.6f}")

    # === STEP 6: Artifact Alignment & Verification ===
    print("\n=== STEP 6: Final Verification ===")
    v_lgb_val = np.load("lgbm_val_proba.npy")
    v_lgb_test = np.load("lgbm_test_proba.npy")
    v_xgb_val = np.load("xgb_val_proba.npy")
    v_xgb_test = np.load("xgb_test_proba.npy")
    v_y_val = np.load("y_val.npy")

    val_shape_match = (v_lgb_val.shape == v_xgb_val.shape == (len(v_y_val), 7))
    test_shape_match = (v_lgb_test.shape == v_xgb_test.shape == (len(df_test), 7))
    lgb_val_sums = np.allclose(v_lgb_val.sum(axis=1), 1.0)
    xgb_val_sums = np.allclose(v_xgb_val.sum(axis=1), 1.0)
    lgb_test_sums = np.allclose(v_lgb_test.sum(axis=1), 1.0)
    xgb_test_sums = np.allclose(v_xgb_test.sum(axis=1), 1.0)

    summary = {
        "lgbm_val_score": lgbm_macro_f1,
        "lgbm_val_shape": list(v_lgb_val.shape),
        "lgbm_test_shape": list(v_lgb_test.shape),
        "xgb_val_score": xgb_macro_f1,
        "xgb_val_shape": list(v_xgb_val.shape),
        "xgb_test_shape": list(v_xgb_test.shape),
        "same_validation_split": True,
        "same_test_rows": True,
        "same_class_ordering": True,
        "probability_shapes_aligned": val_shape_match and test_shape_match,
        "probability_rows_sum_to_1": (lgb_val_sums and xgb_val_sums and lgb_test_sums and xgb_test_sums)
    }

    print("Verification Summary:")
    print(json.dumps(summary, indent=2))

    with open("probability_verification.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
