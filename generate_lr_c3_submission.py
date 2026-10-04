"""Train the Phase-4 best LR (C=3) on full train.csv and write submission_lr_c3.csv.

Does not overwrite submission_lr.csv or any P3.5 files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
OUT = ROOT / "submission_lr_c3.csv"
FEATURES = [f"f{i}" for i in range(1, 175)]


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    test = pd.read_csv(ROOT / "test.csv")
    le = LabelEncoder()
    y = le.fit_transform(train["target"])
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(train[FEATURES]))
    X_te = sc.transform(imp.transform(test[FEATURES]))
    clf = LogisticRegression(
        C=3.0,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        random_state=SEED,
    )
    clf.fit(X_tr, y)
    pred = le.inverse_transform(clf.predict(X_te))
    sub = pd.DataFrame({"id": test["id"], "target": pred})
    assert len(sub) == len(test)
    assert sub["id"].tolist() == test["id"].tolist()
    assert set(sub["target"].unique()).issubset(set(le.classes_))
    assert int(sub.isna().sum().sum()) == 0
    sub.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  shape={sub.shape}")
    print(sub["target"].value_counts().sort_index().to_string())
    print(sub.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
