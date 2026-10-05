import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


ROOT = Path(".")
OUT = ROOT / "artifacts" / "incumbent_081534"
CLASSES = list(range(1, 8))


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def dist(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()}


def validate_submission(path: Path, test: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    assert len(sub) == len(test) == 97731
    assert list(sub.columns) == ["id", "target"]
    assert sub["id"].equals(test["id"])
    assert sub["target"].isin(CLASSES).all()
    assert sub["target"].notna().all()
    return {"file": str(path), "sha16": sha16(path), "rows": len(sub), "distribution": dist(sub)}


def load_proba(path: str, n_rows: int):
    arr = np.load(path)
    assert arr.shape == (n_rows, 7), f"{path} shape {arr.shape}"
    assert np.isfinite(arr).all(), f"{path} has non-finite values"
    assert np.allclose(arr.sum(axis=1), 1, atol=1e-4), f"{path} rows do not sum to 1"
    return arr


def model_report(y, proba):
    pred = proba.argmax(1)
    per = precision_recall_fscore_support(y, pred, labels=np.arange(7), zero_division=0)[2]
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class_f1": {str(i + 1): float(v) for i, v in enumerate(per)},
    }


def write_submission(name: str, proba: np.ndarray, test: pd.DataFrame) -> dict:
    pred = proba.argmax(1) + 1
    sub = pd.DataFrame({"id": test["id"], "target": pred})
    path = ROOT / name
    sub.to_csv(path, index=False)
    return validate_submission(path, test)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_submissions").mkdir(exist_ok=True)
    (OUT / "probability_references").mkdir(exist_ok=True)

    test = pd.read_csv("test.csv", usecols=["id"])
    submissions = []
    for path in sorted(ROOT.glob("submission*.csv")):
        info = validate_submission(path, test)
        submissions.append(info)
        shutil.copy2(path, OUT / "candidate_submissions" / path.name)

    # No metadata file directly identifies the 0.81534 file. Preserve the most
    # suggestively named candidate and all submissions to avoid risking it.
    likely_incumbent = ROOT / "submission_mlp128_xgb45_lr15.csv"
    incumbent_info = next((s for s in submissions if s["file"] == str(likely_incumbent)), None)

    proba_files = [
        "lgbm_test_proba.npy", "xgb_test_proba.npy", "cb_test_proba.npy",
        "ensemble_test_proba.npy", "best_3way_test_proba.npy",
        "phase4b_best_3way_test_proba.npy",
        "next_lgbm_test_proba.npy", "next_xgb_test_proba.npy", "next_cb_test_proba.npy",
    ]
    proba_summary = []
    for name in proba_files:
        path = ROOT / name
        if path.exists():
            arr = load_proba(name, 97731)
            proba_summary.append({"file": name, "shape": list(arr.shape), "sha16": sha16(path)})
            shutil.copy2(path, OUT / "probability_references" / path.name)

    audit = {
        "lb_score": 0.81534,
        "exact_recipe_reconstructed": False,
        "incumbent_file": str(likely_incumbent) if likely_incumbent.exists() else None,
        "incumbent_confidence": "LOW",
        "reason": (
            "No file or metadata explicitly maps 0.81534 to a submission. "
            "submission_mlp128_xgb45_lr15.csv is the most suggestive filename, "
            "but this is an inference, not proof."
        ),
        "incumbent_submission": incumbent_info,
        "all_submission_candidates": submissions,
        "probability_artifacts": proba_summary,
        "oof_artifacts": [
            str(p) for p in sorted(ROOT.rglob("oof_proba.npy"))
            if tuple(np.load(p, mmap_mode="r").shape) == (228039, 7)
        ],
    }
    (OUT / "incumbent_audit.json").write_text(json.dumps(audit, indent=2))

    y = np.load("y_val.npy")
    val_names = {
        "lgbm": "lgbm_val_proba.npy",
        "xgb": "xgb_val_proba.npy",
        "catboost": "cb_val_proba.npy",
        "ensemble": "ensemble_val_proba.npy",
        "best_3way": "best_3way_val_proba.npy",
        "next_lgbm": "next_lgbm_val_proba.npy",
        "next_xgb": "next_xgb_val_proba.npy",
        "next_catboost": "next_cb_val_proba.npy",
    }
    test_names = {k: v.replace("_val_", "_test_") for k, v in val_names.items()}
    test_names["catboost"] = "cb_test_proba.npy"
    test_names["next_catboost"] = "next_cb_test_proba.npy"

    val = {k: load_proba(v, 45608) for k, v in val_names.items() if Path(v).exists()}
    tst = {k: load_proba(test_names[k], 97731) for k in val.keys() if Path(test_names[k]).exists()}
    val = {k: v for k, v in val.items() if k in tst}

    individual = {k: model_report(y, p) for k, p in val.items()}

    # Cheap conservative grids only. No offsets.
    blend_rows = []
    best = {"name": None, "score": -1, "weights": None, "proba": None, "test": None}

    names = ["lgbm", "xgb", "catboost"]
    if all(k in val for k in names):
        for wl in np.arange(0.50, 0.951, 0.025):
            for wx in np.arange(0.0, 0.401, 0.025):
                wc = 1.0 - wl - wx
                if wc < -1e-12:
                    continue
                p = wl * val["lgbm"] + wx * val["xgb"] + wc * val["catboost"]
                score = float(f1_score(y, p.argmax(1), average="macro"))
                row = {"blend": "old_lgbm_xgb_cb", "lgbm": float(wl), "xgb": float(wx), "catboost": float(wc), "macro_f1": score}
                blend_rows.append(row)
                if score > best["score"]:
                    best = {
                        "name": "old_lgbm_xgb_cb",
                        "score": score,
                        "weights": row,
                        "proba": p,
                        "test": wl * tst["lgbm"] + wx * tst["xgb"] + wc * tst["catboost"],
                    }

    names = ["next_lgbm", "next_xgb", "next_catboost"]
    if all(k in val for k in names):
        for wl in np.arange(0.50, 0.951, 0.025):
            for wx in np.arange(0.0, 0.401, 0.025):
                wc = 1.0 - wl - wx
                if wc < -1e-12:
                    continue
                p = wl * val["next_lgbm"] + wx * val["next_xgb"] + wc * val["next_catboost"]
                score = float(f1_score(y, p.argmax(1), average="macro"))
                row = {"blend": "next_lgbm_xgb_cb", "next_lgbm": float(wl), "next_xgb": float(wx), "next_catboost": float(wc), "macro_f1": score}
                blend_rows.append(row)
                if score > best["score"]:
                    best = {
                        "name": "next_lgbm_xgb_cb",
                        "score": score,
                        "weights": row,
                        "proba": p,
                        "test": wl * tst["next_lgbm"] + wx * tst["next_xgb"] + wc * tst["next_catboost"],
                    }

    blend_table = pd.DataFrame(blend_rows).sort_values("macro_f1", ascending=False)
    blend_table.to_csv(OUT / "cheap_blend_grid.csv", index=False)

    candidate_info = None
    if best["test"] is not None:
        candidate_info = write_submission("submission_final_blend.csv", best["test"], test)
        np.save(OUT / "submission_final_blend_test_proba.npy", best["test"])

    final_report = {
        "incumbent": audit,
        "individual_validation": individual,
        "best_cheap_blend": {
            "name": best["name"],
            "validation_macro_f1": best["score"],
            "weights": best["weights"],
            "submission": candidate_info,
        },
        "recommendation": (
            "KEEP INCUMBENT unless manual leaderboard submission slots allow a separate candidate; "
            "the best cheap blend improves old local validation but does not reconstruct/prove superiority "
            "over the stated 0.81534 incumbent."
        ),
    }
    (OUT / "final_push_report.json").write_text(json.dumps(final_report, indent=2))
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
