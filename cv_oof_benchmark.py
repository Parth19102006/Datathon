import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
N_CLASSES = 7
OUT = Path("artifacts/oof")


def set_seed(seed=SEED):
    np.random.seed(seed)


def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train_features = [c for c in train.columns if c not in ("id", "target")]
    test_features = [c for c in test.columns if c != "id"]
    assert train_features == test_features
    assert "id" not in train_features
    X = train[train_features].to_numpy(dtype=np.float32)
    X_test = test[train_features].to_numpy(dtype=np.float32)
    y = train["target"].to_numpy(dtype=np.int64) - 1
    assert sorted(np.unique(y).tolist()) == list(range(N_CLASSES))
    return train, test, train_features, X, y, X_test


def exact_folds(y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    expected = np.empty(len(y), dtype=np.int8)
    dummy = np.zeros(len(y), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf.split(dummy, y)):
        expected[val_idx] = fold

    root_folds = Path("folds.npy")
    if root_folds.exists():
        folds = np.load(root_folds)
        if folds.shape == expected.shape and np.array_equal(folds, expected):
            pass
        else:
            # Do not overwrite the root file silently; this benchmark uses the exact requested folds.
            folds = expected
    else:
        folds = expected
        np.save(root_folds, folds)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "folds.npy", folds)
    assert set(np.unique(folds).tolist()) == set(range(5))
    assert folds.shape == (len(y),)
    return folds


def dataset_stats(train, test, features):
    train_missing_by_col = train[features].isna().sum()
    test_missing_by_col = test[features].isna().sum()
    stats = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_features": int(len(features)),
        "class_counts": {str(k): int(v) for k, v in train["target"].value_counts().sort_index().items()},
        "train_missing_total": int(train_missing_by_col.sum()),
        "test_missing_total": int(test_missing_by_col.sum()),
        "train_missing_min_by_feature": int(train_missing_by_col.min()),
        "train_missing_max_by_feature": int(train_missing_by_col.max()),
        "test_missing_min_by_feature": int(test_missing_by_col.min()),
        "test_missing_max_by_feature": int(test_missing_by_col.max()),
        "duplicate_feature_rows": int(train[features].duplicated().sum()),
        "duplicate_full_train_rows": int(train.duplicated().sum()),
        "constant_columns": [c for c in features if train[c].nunique(dropna=False) <= 1],
        "near_constant_columns_99_9pct": [
            c for c in features
            if train[c].value_counts(dropna=False, normalize=True).iloc[0] >= 0.999
        ],
        "dtypes": {str(k): int(v) for k, v in train[features].dtypes.value_counts().items()},
    }
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def balanced_sample_weight(y, exponent=1.0, normalize=False):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: (total / (len(classes) * cnt)) ** exponent for c, cnt in zip(classes, counts)}
    out = np.array([weights[c] for c in y], dtype=np.float32)
    if normalize:
        out /= out.mean()
    return out


def make_model(name):
    if name == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=10.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight=None,
                n_jobs=-1,
                random_state=SEED,
            ),
        )
    if name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128,),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=1e-3,
                max_iter=30,
                early_stopping=False,
                random_state=SEED,
                verbose=False,
            ),
        )
    if name == "xgb":
        from xgboost import XGBClassifier

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                objective="multi:softprob",
                num_class=N_CLASSES,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                n_estimators=500,
                reg_alpha=0.1,
                reg_lambda=1.5,
                tree_method="hist",
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=SEED,
            ),
        )
    if name == "lgbm":
        import lightgbm as lgb

        return make_pipeline(
            SimpleImputer(strategy="median"),
            lgb.LGBMClassifier(
                objective="multiclass",
                num_class=N_CLASSES,
                learning_rate=0.07921743455371703,
                n_estimators=500,
                num_leaves=30,
                max_depth=8,
                min_child_samples=23,
                subsample=0.7741721785653952,
                colsample_bytree=0.6521351677938568,
                reg_alpha=0.4694209700587524,
                reg_lambda=8.513585349506269e-05,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
                verbose=-1,
            ),
        )
    raise ValueError(name)


def fit_model(name, model, X_tr, y_tr):
    if name == "xgb":
        model.fit(X_tr, y_tr, xgbclassifier__sample_weight=balanced_sample_weight(y_tr))
    else:
        model.fit(X_tr, y_tr)
    return model


def evaluate(y, proba):
    pred = proba.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(N_CLASSES), zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": {str(i): float(v) for i, v in enumerate(precision)},
        "recall": {str(i): float(v) for i, v in enumerate(recall)},
        "f1": {str(i): float(v) for i, v in enumerate(f1)},
        "support": {str(i): int(v) for i, v in enumerate(support)},
        "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(N_CLASSES)).astype(int).tolist(),
        "prediction_distribution": {str(i): int(v) for i, v in zip(*np.unique(pred, return_counts=True))},
        "class_0_recall": float(recall[0]),
        "class_1_recall": float(recall[1]),
        "predicted_positive_percentage": None,
        "threshold_note": "N/A for 7-class single-label argmax; no binary threshold applied.",
    }


def train_cv_model(name, X, y, X_test, folds):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    oof = np.zeros((len(y), N_CLASSES), dtype=np.float32)
    test_proba = np.zeros((len(X_test), N_CLASSES), dtype=np.float64)
    fold_scores = []

    for fold in range(5):
        fold_start = time.time()
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]
        model = make_model(name)
        fit_model(name, model, X[tr_idx], y[tr_idx])
        va_proba = model.predict_proba(X[va_idx]).astype(np.float32)
        te_proba = model.predict_proba(X_test).astype(np.float64)
        assert va_proba.shape == (len(va_idx), N_CLASSES)
        assert te_proba.shape == (len(X_test), N_CLASSES)
        assert np.isfinite(va_proba).all() and np.isfinite(te_proba).all()
        assert np.allclose(va_proba.sum(axis=1), 1, atol=1e-4)
        assert np.allclose(te_proba.sum(axis=1), 1, atol=1e-4)
        oof[va_idx] = va_proba
        test_proba += te_proba / 5.0
        score = f1_score(y[va_idx], va_proba.argmax(axis=1), average="macro")
        fold_scores.append(float(score))
        print(f"{name} fold {fold}: macro_f1={score:.6f} seconds={time.time() - fold_start:.1f}", flush=True)

    assert oof.shape == (228039, N_CLASSES)
    assert test_proba.shape == (97731, N_CLASSES)
    assert np.isfinite(oof).all() and np.isfinite(test_proba).all()
    assert np.allclose(oof.sum(axis=1), 1, atol=1e-4)
    assert np.allclose(test_proba.sum(axis=1), 1, atol=1e-4)

    np.save(OUT / f"{name}_oof_proba.npy", oof)
    np.save(OUT / f"{name}_test_proba.npy", test_proba.astype(np.float32))
    metrics = evaluate(y, oof)
    metrics.update({
        "model": name,
        "fold_macro_f1": fold_scores,
        "fold_mean_macro_f1": float(np.mean(fold_scores)),
        "fold_std_macro_f1": float(np.std(fold_scores)),
        "runtime_seconds": float(time.time() - start),
        "threshold_0_5_macro_f1": None,
        "best_threshold": None,
        "best_threshold_macro_f1": None,
    })
    (OUT / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, oof, test_proba


def diversity(y, probas):
    rows = []
    names = list(probas)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = probas[a], probas[b]
            pred_a, pred_b = pa.argmax(axis=1), pb.argmax(axis=1)
            rows.append({
                "model_a": a,
                "model_b": b,
                "probability_correlation": float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1]),
                "prediction_agreement": float(np.mean(pred_a == pred_b)),
                "prediction_disagreement": float(np.mean(pred_a != pred_b)),
                "a_correct_b_wrong": int(np.sum((pred_a == y) & (pred_b != y))),
                "b_correct_a_wrong": int(np.sum((pred_b == y) & (pred_a != y))),
            })
    pd.DataFrame(rows).to_csv(OUT / "model_diversity.csv", index=False)
    return rows


def simple_blends(y, probas):
    rows = []
    names_by_score = sorted(probas, key=lambda n: f1_score(y, probas[n].argmax(1), average="macro"), reverse=True)
    candidates = []
    strongest = names_by_score[0]
    candidates.append((f"{strongest}_alone", {strongest: 1.0}))
    if len(names_by_score) >= 2:
        a, b = names_by_score[:2]
        candidates.extend([
            (f"{a}_{b}_50_50", {a: 0.5, b: 0.5}),
            (f"{a}_{b}_70_30", {a: 0.7, b: 0.3}),
            (f"{a}_{b}_80_20", {a: 0.8, b: 0.2}),
        ])
    useful = names_by_score[: min(4, len(names_by_score))]
    candidates.append(("equal_weight_useful", {n: 1 / len(useful) for n in useful}))

    best = None
    for label, weights in candidates:
        p = np.zeros_like(next(iter(probas.values())), dtype=np.float64)
        for n, w in weights.items():
            p += w * probas[n]
        score = float(f1_score(y, p.argmax(1), average="macro"))
        row = {"blend": label, "macro_f1": score, "weights": weights}
        rows.append(row)
        if best is None or score > best["macro_f1"]:
            best = row
    (OUT / "simple_blends.json").write_text(json.dumps(rows, indent=2))
    return rows, best


def write_report(stats, model_configs, metrics, div_rows, blend_rows, best_blend):
    lines = []
    lines.append("# Clean 5-Fold OOF Benchmark")
    lines.append("")
    lines.append("## CURRENT INCUMBENT")
    lines.append("")
    lines.append("Leaderboard: `0.81534`")
    lines.append("")
    lines.append("Local reproduction: `UNKNOWN`")
    lines.append("")
    lines.append("Do not interpret OOF as leaderboard replacement evidence.")
    lines.append("")
    lines.append("## Dataset Statistics")
    lines.append("")
    lines.append(f"- Train rows: `{stats['train_rows']}`")
    lines.append(f"- Test rows: `{stats['test_rows']}`")
    lines.append(f"- Number of features: `{stats['n_features']}`")
    lines.append(f"- Class counts: `{stats['class_counts']}`")
    lines.append(f"- Train missing total: `{stats['train_missing_total']}`")
    lines.append(f"- Test missing total: `{stats['test_missing_total']}`")
    lines.append(f"- Duplicate feature rows: `{stats['duplicate_feature_rows']}`")
    lines.append(f"- Duplicate full train rows: `{stats['duplicate_full_train_rows']}`")
    lines.append(f"- Constant columns: `{stats['constant_columns']}`")
    lines.append(f"- Near-constant columns >=99.9%: `{stats['near_constant_columns_99_9pct']}`")
    lines.append("")
    lines.append("## Fold Configuration")
    lines.append("")
    lines.append("`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`")
    lines.append("")
    lines.append("Fold assignments saved to `artifacts/oof/folds.npy`.")
    lines.append("")
    lines.append("## Model Configurations")
    lines.append("")
    for name, config in model_configs.items():
        lines.append(f"- `{name}`: {config}")
    lines.append("")
    lines.append("## OOF RESULTS")
    lines.append("")
    lines.append("| Model | OOF F1 @ argmax | Best OOF F1 | Best threshold | Accuracy | Class 0 Recall | Class 1 Recall | Positive % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, m in sorted(metrics.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True):
        lines.append(
            f"| {name} | {m['macro_f1']:.6f} | N/A | N/A | {m['accuracy']:.6f} | "
            f"{m['class_0_recall']:.6f} | {m['class_1_recall']:.6f} | N/A |"
        )
    lines.append("")
    lines.append("Threshold note: this is a 7-class single-label task, so binary threshold and positive percentage are not applicable. Predictions use probability argmax.")
    lines.append("")
    lines.append("## Probability Correlations And Prediction Disagreement")
    lines.append("")
    lines.append("| Model A | Model B | Probability Corr | Agreement | Disagreement | A Correct/B Wrong | B Correct/A Wrong |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in div_rows:
        lines.append(
            f"| {r['model_a']} | {r['model_b']} | {r['probability_correlation']:.6f} | "
            f"{r['prediction_agreement']:.6f} | {r['prediction_disagreement']:.6f} | "
            f"{r['a_correct_b_wrong']} | {r['b_correct_a_wrong']} |"
        )
    lines.append("")
    strongest = max(metrics, key=lambda n: metrics[n]["macro_f1"])
    lines.append("## Simple Blend Results")
    lines.append("")
    lines.append("| Blend | Macro F1 | Weights |")
    lines.append("| --- | ---: | --- |")
    for r in sorted(blend_rows, key=lambda x: x["macro_f1"], reverse=True):
        lines.append(f"| {r['blend']} | {r['macro_f1']:.6f} | `{r['weights']}` |")
    lines.append("")
    lines.append("## BEST OOF CANDIDATE")
    lines.append("")
    best_model = max(metrics, key=lambda n: metrics[n]["macro_f1"])
    lines.append(f"Model: `{best_model}`")
    lines.append(f"Macro F1: `{metrics[best_model]['macro_f1']:.6f}`")
    lines.append("Threshold: `N/A for multiclass argmax`")
    lines.append("")
    lines.append("## BLEND RESULTS")
    lines.append("")
    lines.append(f"Best simple blend: `{best_blend['blend']}`")
    lines.append(f"Macro F1: `{best_blend['macro_f1']:.6f}`")
    lines.append(f"Weights: `{best_blend['weights']}`")
    lines.append("")
    lines.append("## Comparison Against Incumbent")
    lines.append("")
    lines.append("The incumbent is a leaderboard score (`0.81534`) with unknown local reproduction. This OOF benchmark is for model comparison only; it does not prove leaderboard improvement.")
    (OUT / "oof_benchmark_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lr", "xgb", "mlp", "lgbm"])
    args = parser.parse_args()
    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)

    train, test, features, X, y, X_test = load_data()
    stats = dataset_stats(train, test, features)
    folds = exact_folds(y)

    model_configs = {
        "lr": "median imputer + StandardScaler + LogisticRegression(C=10, lbfgs, max_iter=1000)",
        "xgb": "median imputer + XGBClassifier(hist, 500 trees, balanced sample weights)",
        "mlp": "median imputer + StandardScaler + MLPClassifier(128 hidden units, max_iter=30)",
        "lgbm": "median imputer + LightGBM balanced, existing strong holdout params, no early stopping",
    }

    metrics = {}
    probas = {}
    for name in args.models:
        m, oof, _ = train_cv_model(name, X, y, X_test, folds)
        metrics[name] = m
        probas[name] = oof

    div_rows = diversity(y, probas)
    blend_rows, best_blend = simple_blends(y, probas)
    write_report(stats, {k: model_configs[k] for k in args.models}, metrics, div_rows, blend_rows, best_blend)
    print(f"Report written to {OUT / 'oof_benchmark_report.md'}", flush=True)


if __name__ == "__main__":
    main()
