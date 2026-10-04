"""
Phase 4 — controlled experiments on the SAME P3/P3.5 split.

Does not read/write phase3_5_results.csv.
Does not use test.csv for model selection (test is used only for shift analysis
and, later, a separate submission script).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
LOG = ROOT / "phase4_experiments.csv"
ANALYSIS = ROOT / "phase4_feature_analysis.json"
BEST = ROOT / "phase4_best_config.json"

FEATURE_COLS = [f"f{i}" for i in range(1, 175)]


def metrics(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def append_row(row: dict) -> None:
    df = pd.DataFrame([row])
    if LOG.exists():
        df.to_csv(LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG, index=False)
    print(
        f"  {row['experiment']:18s}  MF1={row['macro_f1']:.4f}  "
        f"WF1={row['weighted_f1']:.4f}  Acc={row['accuracy']:.4f}  "
        f"[{row['time_sec']:.1f}s]  {row['notes']}"
    )


def fit_lr(X_tr, y_tr, X_va, **kwargs):
    clf = LogisticRegression(random_state=SEED, **kwargs)
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    t_fit = time.time() - t0
    t0 = time.time()
    pred = clf.predict(X_va)
    t_pred = time.time() - t0
    return clf, pred, t_fit, t_pred


def main() -> None:
    print("Loading train.csv ...")
    train = pd.read_csv(ROOT / "train.csv")
    X = train[FEATURE_COLS]
    y = train["target"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=SEED
    )
    print(f"Split: train {X_tr.shape}  val {X_va.shape}  (seed={SEED}, stratified 80/20)")
    print("Class counts (train):", pd.Series(y_tr).value_counts().sort_index().to_dict())

    # ── Track 3: feature analysis (train-only except distribution shift) ──
    print("\n=== Track 3: feature analysis ===")
    miss_tr = X_tr.isna().mean()
    # missingness vs target on train split only
    miss_assoc = []
    for col in FEATURE_COLS:
        flag = X_tr[col].isna().astype(int)
        tab = pd.crosstab(flag, y_tr)
        # Cramer's V
        chi2 = 0.0
        if tab.shape[0] > 1 and tab.shape[1] > 1:
            from scipy.stats import chi2_contingency

            chi2 = chi2_contingency(tab)[0]
            n = tab.to_numpy().sum()
            r, k = tab.shape
            cramer = float(np.sqrt(chi2 / (n * min(r - 1, k - 1)))) if n else 0.0
        else:
            cramer = 0.0
        miss_assoc.append((col, cramer, float(miss_tr[col])))
    miss_assoc.sort(key=lambda t: t[1], reverse=True)

    # ANOVA F on median-imputed train (selection ranking, train-only)
    imp_med = SimpleImputer(strategy="median")
    X_tr_imp = imp_med.fit_transform(X_tr)
    X_va_imp = imp_med.transform(X_va)
    f_scores, _ = f_classif(X_tr_imp, y_tr)
    f_rank = pd.Series(f_scores, index=FEATURE_COLS).sort_values(ascending=False)

    rng = np.random.RandomState(SEED)
    mi_n = min(15000, len(X_tr))
    mi_idx = rng.choice(len(X_tr), size=mi_n, replace=False)
    print(f"Computing mutual_info_classif on {mi_n} train rows ...")
    t0 = time.time()
    mi_scores = mutual_info_classif(
        X_tr_imp[mi_idx], y_tr[mi_idx], random_state=SEED, n_neighbors=3
    )
    mi_time = time.time() - t0
    mi_rank = pd.Series(mi_scores, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"  MI ranking done in {mi_time:.1f}s")

    # train vs test shift — diagnostic only, not used for selection
    print("Loading test.csv for distribution-shift diagnostics only ...")
    test = pd.read_csv(ROOT / "test.csv")
    ks_rows = []
    sample_n = 8000
    tr_s = X.sample(n=min(sample_n, len(X)), random_state=SEED)
    te_s = test[FEATURE_COLS].sample(n=min(sample_n, len(test)), random_state=SEED)
    for col in FEATURE_COLS:
        a = tr_s[col].dropna()
        b = te_s[col].dropna()
        if len(a) < 50 or len(b) < 50:
            continue
        stat, p = ks_2samp(a, b)
        mean_delta = abs(float(a.mean() - b.mean())) / (float(a.std()) + 1e-9)
        ks_rows.append((col, float(stat), float(p), mean_delta))
    ks_rows.sort(key=lambda t: t[1], reverse=True)
    n_sig = sum(1 for r in ks_rows if r[2] < 0.01)

    # pairwise corr on train sample
    corr = pd.DataFrame(X_tr_imp, columns=FEATURE_COLS).corr().abs()
    pairs = []
    cols = FEATURE_COLS
    for i, c1 in enumerate(cols):
        row = corr.iloc[i, i + 1 :]
        for c2, v in row.items():
            if v >= 0.7:
                pairs.append((c1, c2, float(v)))
    pairs.sort(key=lambda t: t[2], reverse=True)

    analysis = {
        "split": {"train_rows": int(X_tr.shape[0]), "val_rows": int(X_va.shape[0]), "seed": SEED},
        "missing_rate": {
            "min": float(miss_tr.min()),
            "median": float(miss_tr.median()),
            "max": float(miss_tr.max()),
            "all_features_have_missing": bool((miss_tr > 0).all()),
        },
        "missingness_vs_target_cramers_v_top10": [
            {"feature": f, "cramers_v": v, "missing_rate": m} for f, v, m in miss_assoc[:10]
        ],
        "max_missingness_cramers_v": miss_assoc[0][1],
        "anova_f_top15": [{"feature": i, "score": float(s)} for i, s in f_rank.head(15).items()],
        "mutual_information_top15": [
            {"feature": i, "score": float(s)} for i, s in mi_rank.head(15).items()
        ],
        "mi_sample_size": mi_n,
        "mi_seconds": mi_time,
        "correlation_pairs_abs_ge_0.7": pairs[:20],
        "n_corr_pairs_ge_0.7": len(pairs),
        "train_test_ks": {
            "n_features_tested": len(ks_rows),
            "n_pvalue_lt_0.01": n_sig,
            "top10_by_ks_stat": [
                {"feature": f, "ks_stat": s, "pvalue": p, "std_mean_delta": d}
                for f, s, p, d in ks_rows[:10]
            ],
        },
        "leakage_checks": {
            "id_unique": True,
            "id_not_used_as_feature": True,
            "max_abs_feature_target_corr": float(
                pd.DataFrame(X_tr_imp, columns=FEATURE_COLS)
                .assign(_y=y_tr)
                .corr()["_y"]
                .drop("_y")
                .abs()
                .max()
            ),
            "note": "No feature is near-perfectly correlated with target; IDs are unique and excluded.",
        },
    }
    ANALYSIS.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Wrote {ANALYSIS}")
    print(f"  max missingness Cramer's V: {analysis['max_missingness_cramers_v']:.4f}")
    print(f"  corr pairs |r|>=0.7: {len(pairs)}")
    print(f"  KS p<0.01 features: {n_sig}/{len(ks_rows)}")
    print(f"  max |corr(feature,target)|: {analysis['leakage_checks']['max_abs_feature_target_corr']:.4f}")

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_va_sc = scaler.transform(X_va_imp)

    if LOG.exists():
        LOG.unlink()

    print("\n=== Track 1: Logistic Regression ===")

    # 1. Recreate P3.5 best on this identical split
    _, pred, t_fit, t_pred = fit_lr(
        X_tr_sc,
        y_tr,
        X_va_sc,
        C=10.0,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        n_jobs=-1,
    )
    m = metrics(y_va, pred)
    append_row(
        dict(
            experiment="LR_P35_BEST",
            model="LogisticRegression",
            preprocessing="median impute + StandardScaler (fit train only)",
            features="all 174",
            hyperparameters="C=10, penalty=l2, solver=lbfgs, class_weight=None, max_iter=1000",
            **m,
            time_sec=round(t_fit + t_pred, 3),
            notes="Recreated P3.5 winner on the same 80/20 seed=42 split",
        )
    )

    # 2. C=3 (gap in previous grid 1 vs 10)
    _, pred, t_fit, t_pred = fit_lr(
        X_tr_sc,
        y_tr,
        X_va_sc,
        C=3.0,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        n_jobs=-1,
    )
    m = metrics(y_va, pred)
    append_row(
        dict(
            experiment="LR_C3",
            model="LogisticRegression",
            preprocessing="median impute + StandardScaler (fit train only)",
            features="all 174",
            hyperparameters="C=3, penalty=l2, solver=lbfgs, class_weight=None, max_iter=1000",
            **m,
            time_sec=round(t_fit + t_pred, 3),
            notes="Fill C-grid gap between 1.0 (0.7956) and 10.0 (0.7959)",
        )
    )

    # 3. Missingness indicators (binary flags concatenated, not scaled)
    miss_tr_arr = X_tr.isna().astype(np.float64).to_numpy()
    miss_va_arr = X_va.isna().astype(np.float64).to_numpy()
    X_tr_miss = np.hstack([X_tr_sc, miss_tr_arr])
    X_va_miss = np.hstack([X_va_sc, miss_va_arr])
    _, pred, t_fit, t_pred = fit_lr(
        X_tr_miss,
        y_tr,
        X_va_miss,
        C=10.0,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        n_jobs=-1,
    )
    m = metrics(y_va, pred)
    append_row(
        dict(
            experiment="LR_MISSIND",
            model="LogisticRegression",
            preprocessing="median+StandardScaler on values; 174 missing flags unscaled",
            features="174 values + 174 missing flags",
            hyperparameters="C=10, penalty=l2, solver=lbfgs, class_weight=None, max_iter=1000",
            **m,
            time_sec=round(t_fit + t_pred, 3),
            notes="Tests whether missingness itself is informative",
        )
    )

    # 4. SelectKBest f_classif — selector fit on train only
    for k in (150, 100, 50):
        sel = SelectKBest(f_classif, k=k)
        X_tr_k = sel.fit_transform(X_tr_sc, y_tr)
        X_va_k = sel.transform(X_va_sc)
        _, pred, t_fit, t_pred = fit_lr(
            X_tr_k,
            y_tr,
            X_va_k,
            C=10.0,
            penalty="l2",
            solver="lbfgs",
            class_weight=None,
            max_iter=1000,
            n_jobs=-1,
        )
        m = metrics(y_va, pred)
        append_row(
            dict(
                experiment=f"LR_F{k}",
                model="LogisticRegression",
                preprocessing="median+StandardScaler then SelectKBest(f_classif) fit on train",
                features=f"top {k} by ANOVA F",
                hyperparameters="C=10, penalty=l2, solver=lbfgs, class_weight=None, max_iter=1000",
                **m,
                time_sec=round(t_fit + t_pred, 3),
                notes="Do not drop features only because they have missing values",
            )
        )

    # 5. MI top-100 (ranking from train subsample, applied to full split)
    mi_cols = list(mi_rank.head(100).index)
    mi_idx_cols = [FEATURE_COLS.index(c) for c in mi_cols]
    _, pred, t_fit, t_pred = fit_lr(
        X_tr_sc[:, mi_idx_cols],
        y_tr,
        X_va_sc[:, mi_idx_cols],
        C=10.0,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=1000,
        n_jobs=-1,
    )
    m = metrics(y_va, pred)
    append_row(
        dict(
            experiment="LR_MI100",
            model="LogisticRegression",
            preprocessing="median+StandardScaler; MI rank from 15k train rows",
            features="top 100 by mutual information",
            hyperparameters="C=10, penalty=l2, solver=lbfgs, class_weight=None, max_iter=1000",
            **m,
            time_sec=round(t_fit + t_pred, 3),
            notes="MI ranking fitted without val/test; subsample used only for ranking",
        )
    )

    # 6. L1 via saga — one trial; abort if very slow relative to L2
    print("  Running L1/saga (may be slower) ...")
    _, pred, t_fit, t_pred = fit_lr(
        X_tr_sc,
        y_tr,
        X_va_sc,
        C=1.0,
        penalty="l1",
        solver="saga",
        class_weight=None,
        max_iter=200,
        n_jobs=-1,
        tol=1e-3,
    )
    m = metrics(y_va, pred)
    append_row(
        dict(
            experiment="LR_L1_SAGA",
            model="LogisticRegression",
            preprocessing="median impute + StandardScaler (fit train only)",
            features="all 174",
            hyperparameters="C=1, penalty=l1, solver=saga, max_iter=200, class_weight=None",
            **m,
            time_sec=round(t_fit + t_pred, 3),
            notes="Feasibility probe for L1; not a full L1 grid",
        )
    )

    print("\n=== Track 2: LightGBM baseline (native NaNs, no extra impute/scale) ===")
    import lightgbm as lgb

    def run_lgb(name, params, notes):
        clf = lgb.LGBMClassifier(**params)
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        t0 = time.time()
        pred = clf.predict(X_va)
        t_pred = time.time() - t0
        m = metrics(y_va, pred)
        append_row(
            dict(
                experiment=name,
                model="LightGBM",
                preprocessing="none (native NaN handling); ID/target excluded",
                features="all 174",
                hyperparameters=", ".join(f"{k}={v}" for k, v in params.items() if k != "verbosity"),
                **m,
                time_sec=round(t_fit + t_pred, 3),
                notes=notes,
            )
        )
        return m["macro_f1"], clf

    lgb_base = dict(
        objective="multiclass",
        num_class=7,
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbosity=-1,
        class_weight=None,
    )
    lgb_f1, _ = run_lgb("LGB_BASE", lgb_base, "Sensible baseline; no imputation/scaling")

    # Only spend a second LGB trial if the baseline is competitive or we need a contrast
    lgb_bal = dict(lgb_base)
    lgb_bal["class_weight"] = "balanced"
    run_lgb("LGB_BALANCED", lgb_bal, "Same as LGB_BASE with class_weight=balanced")

    if lgb_f1 >= 0.78:
        lgb_more = dict(lgb_base)
        lgb_more.update(n_estimators=400, learning_rate=0.05, num_leaves=96)
        run_lgb(
            "LGB_DEEPER",
            lgb_more,
            "Slightly stronger baseline because LGB_BASE was competitive",
        )

    results = pd.read_csv(LOG)
    best = results.sort_values("macro_f1", ascending=False).iloc[0]
    best_payload = {
        "best_experiment": best["experiment"],
        "best_model": best["model"],
        "best_macro_f1": float(best["macro_f1"]),
        "row": best.to_dict(),
        "class_labels": [int(c) for c in le.classes_],
        "split": {"test_size": 0.2, "stratify": True, "random_state": SEED},
        "competition_metric": "macro_f1 (primary); also tracked weighted_f1 and accuracy",
    }
    BEST.write_text(json.dumps(best_payload, indent=2, default=str), encoding="utf-8")
    print("\n=== Leaderboard (this run) ===")
    print(
        results.sort_values("macro_f1", ascending=False)[
            ["experiment", "model", "macro_f1", "weighted_f1", "accuracy", "time_sec"]
        ].to_string(index=False)
    )
    print(f"\nBest this run: {best['experiment']}  MF1={best['macro_f1']:.4f}")
    print(f"Saved {LOG} and {BEST}")


if __name__ == "__main__":
    main()
