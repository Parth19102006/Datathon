# DATATHON Phase 0 Inventory

Working directory: `C:/Users/ASUS/OneDrive/Desktop/Datathon`

## Dataset

| File | Purpose | Shape / Columns | Notes |
| --- | --- | --- | --- |
| `train.csv` | Training data | 228,039 rows; 176 columns | Columns are `id`, `f1`..`f174`, `target`. Target labels are `1..7`. |
| `test.csv` | Test data | 97,731 rows; 175 columns | Columns are `id`, `f1`..`f174`. |
| `folds.npy` | Fixed 5-fold assignments | `(228039,)` | Created with stratified 5-fold intent; should be reused, not regenerated casually. |
| `y_val.npy` | Old 80/20 validation labels | `(45608,)` | Belongs to old holdout workflow, not clean 5-fold CV. |
| `class_order.json` | Class order | `[1,2,3,4,5,6,7]` | Confirms probability column order convention. |

Target column: `target`

ID/index column: `id`

Existing notebooks: none found (`*.ipynb` search returned empty).

## Incumbent 0.81534 Audit

Best known leaderboard score: `0.81534`

Exact source: **could not be proven from local files**.

Likely incumbent file: `submission_mlp128_xgb45_lr15.csv`

Confidence: **LOW**

Reason: no script, JSON, Markdown, or notebook explicitly maps `0.81534` to a submitted CSV. The filename `submission_mlp128_xgb45_lr15.csv` is suggestive, but this is inference rather than proof.

Preservation directory: `artifacts/incumbent_081534/`

This directory contains copies of all local `submission*.csv` files and key compatible test probability arrays.

Likely incumbent distribution:

| Class | Count |
| --- | ---: |
| 1 | 11747 |
| 2 | 1197 |
| 3 | 22884 |
| 4 | 22388 |
| 5 | 15018 |
| 6 | 24101 |
| 7 | 396 |

## Existing Submission Files

| File | Purpose / Model Clue | Local validation score | Leaderboard score | Incumbent? |
| --- | --- | ---: | ---: | --- |
| `submission_mlp128_xgb45_lr15.csv` | Filename suggests MLP-128 + XGB 45% + LR 15% blend | unknown | possibly `0.81534`, unproven | likely but unproven |
| `submission_final_blend.csv` | Cheap cached-probability blend from final push | old holdout `0.807114` | unknown | no |
| `submission_next.csv` | Prior next-phase LGBM+XGB+CatBoost with offsets | old holdout `0.807951` | unknown | no |
| `submission_phase4b_multipliers.csv` | LGBM/XGB/CatBoost 3-way with multipliers | old holdout `0.804643` | unknown | no |
| `submission_ensemble.csv` | LGBM/XGB ensemble | old holdout `0.804109` | unknown | no |
| `submission_lgb.csv` | LightGBM full-train submission | holdout source `0.802625` | unknown | no |
| `submission_lr.csv` | Logistic regression submission | `0.795856` | unknown | no |
| `submission_lr_c3.csv` | Logistic regression C=3 variant | around `0.7969` in phase 4 reports | unknown | no |
| `submission (1).csv` | Older benchmark file used by `analyze.py`; comment says 81.162% | unknown | unknown | no proof |
| `submission (5).csv`, `submission (6).csv`, `submission3.csv`, `submission_improved2.csv` | Older/manual candidates | unknown | unknown | no proof |

## Existing Probability Artifacts

| File | Purpose / Model | Shape | Local validation score if known | Incumbent? |
| --- | --- | --- | ---: | --- |
| `lgbm_val_proba.npy`, `lgbm_test_proba.npy` | Old holdout LightGBM | `(45608,7)`, `(97731,7)` | `0.802625` | no proof |
| `xgb_val_proba.npy`, `xgb_test_proba.npy` | Old holdout XGBoost | `(45608,7)`, `(97731,7)` | `0.755838` | no proof |
| `cb_val_proba.npy`, `cb_test_proba.npy` | Old holdout CatBoost | `(45608,7)`, `(97731,7)` | `0.743161` | no proof |
| `ensemble_val_proba.npy`, `ensemble_test_proba.npy` | Old LGBM/XGB ensemble | `(45608,7)`, `(97731,7)` | `0.804109` | no proof |
| `best_3way_val_proba.npy`, `best_3way_test_proba.npy` | Old LGBM/XGB/CatBoost blend | `(45608,7)`, `(97731,7)` | `0.804546` | no proof |
| `phase4b_best_3way_test_proba.npy` | Phase 4B 3-way test probabilities | `(97731,7)` | `0.804643` from weights JSON | no proof |
| `next_lgbm_val_proba.npy`, `next_lgbm_test_proba.npy` | Later saved LGBM probabilities | `(45608,7)`, `(97731,7)` | `0.803290` no offsets | no proof |
| `next_xgb_val_proba.npy`, `next_xgb_test_proba.npy` | Later XGBoost probabilities | `(45608,7)`, `(97731,7)` | `0.799351` | no proof |
| `next_cb_val_proba.npy`, `next_cb_test_proba.npy` | Later CatBoost probabilities | `(45608,7)`, `(97731,7)` | `0.779618` | no proof |
| `artifacts/*/oof_proba.npy` | Existing fast 5-fold OOF LGBM experiments | `(228039,7)` | best existing fast OOF around `0.771843`; `LGBM_balanced_stage0_fast` `0.768062` | no |

## Scripts And Reports

| File | Purpose | Notes |
| --- | --- | --- |
| `common.py` | Shared CV helpers | Contains data loading, feature validation, folds, metrics, and assertions. |
| `run_lgbm_balanced.py` | No-early-stopping 5-fold LGBM balanced runner | Stage-0 fast run exists; not a final candidate. |
| `quick_final_push.py` | Fast audit/preservation/cached-blend script | Created `artifacts/incumbent_081534/` and `submission_final_blend.csv`. |
| `results.md` | Final-push summary | Recommends keeping incumbent unless spare submission slots allow extra candidate. |
| `generate_probability_artifacts.py` | Generates old LGBM/XGB holdout probabilities | Uses old 80/20 validation split. |
| `ensemble_experiment.py` | Old LGBM/XGB blending | Old holdout only. |
| `phase4a_catboost_experiment.py`, `phase4b_optimization.py` | Old 3-way CatBoost/LGBM/XGB experiments | Old holdout only. |
| `next_artifact_pipeline.py` | Later cached-probability pipeline | Uses existing LGBM artifacts and trains XGB/CatBoost; not clean 5-fold. |
| `lgb_cv.py` | Older 5-fold LGBM runner | Regenerates folds internally and used early stopping on scored folds; not compliant with newest clean-CV requirements. |
| `phase3_5_best_models.json` | Best old holdout model configs | LR `0.795856`, HGB `0.780139`, LGBM `0.802625`. |
| `probability_verification.json` | Shape/order verification for old LGBM/XGB probability arrays | Confirms compatible old holdout arrays. |
| `artifacts/incumbent_081534/final_push_report.json` | Incumbent preservation and cheap blend report | Current best audit record. |

## Current State

Phase 0 inventory is complete.

No training was started for this request.

The `0.81534` incumbent should remain untouched. Since the exact winning recipe cannot be proven, preserve all copied submission candidates under `artifacts/incumbent_081534/` and treat `submission_mlp128_xgb45_lr15.csv` only as the likely incumbent, not as verified fact.
