import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import time

def main():
    print("Loading datasets...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    train_features = [c for c in train.columns if c not in ["id", "target"]]
    test_features = [c for c in test.columns if c != "id"]
    
    if train_features != test_features:
        print("ERROR: Features do not match!")
        return
    else:
        print("Feature columns match exactly.")
        
    X_train = train[train_features]
    y_train = train["target"]
    X_test = test[test_features]
    
    print("Applying SimpleImputer(median)...")
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    print("Loading best parameters...")
    with open("phase3_5_best_models.json", "r") as f:
        best_models = json.load(f)
    
    lgb_params = best_models["LightGBM"]["best_params"]
    print("Best params:", lgb_params)
    
    print("Training LightGBM on FULL train dataset...")
    t0 = time.time()
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_train_imp, y_train_enc)
    print(f"Training completed in {time.time()-t0:.2f} seconds.")
    
    print("Predicting on test dataset...")
    preds_enc = clf.predict(X_test_imp)
    preds = le.inverse_transform(preds_enc)
    
    print("Generating submission...")
    sub = pd.DataFrame({"id": test["id"], "target": preds})
    
    # Validation checks
    assert len(sub) == len(test), "Row count mismatch"
    assert (sub["id"] == test["id"]).all(), "IDs do not match exactly"
    assert sub["target"].isna().sum() == 0, "Missing predictions found"
    assert set(sub["target"].unique()).issubset(set(range(1,8))), "Predictions are outside classes 1-7"
    assert list(sub.columns) == ["id", "target"], "Incorrect column names or order"
    
    sub.to_csv("submission_lgb.csv", index=False)
    print("Submission saved to submission_lgb.csv")
    
    # Save test distribution stats to a json for the report
    dist = sub["target"].value_counts(normalize=True).sort_index().to_dict()
    stats = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "prediction_distribution": dist,
        "classes_predicted": list(dist.keys()),
        "lgb_params": lgb_params
    }
    with open("lgb_test_stats.json", "w") as f:
        json.dump(stats, f)

if __name__ == "__main__":
    main()
