# Clean 5-Fold OOF Benchmark

## CURRENT INCUMBENT

Leaderboard: `0.81534`

Local reproduction: `UNKNOWN`

Do not interpret OOF as leaderboard replacement evidence.

## Dataset Statistics

- Train rows: `228039`
- Test rows: `97731`
- Number of features: `174`
- Class counts: `{'1': 27410, '2': 2519, '3': 52970, '4': 51829, '5': 32964, '6': 59550, '7': 797}`
- Train missing total: `5949963`
- Test missing total: `2550057`
- Duplicate feature rows: `0`
- Duplicate full train rows: `0`
- Constant columns: `[]`
- Near-constant columns >=99.9%: `[]`

## Fold Configuration

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

Fold assignments saved to `artifacts/oof/folds.npy`.

## Model Configurations

- `lr`: median imputer + StandardScaler + LogisticRegression(C=10, lbfgs, max_iter=1000)
- `xgb`: median imputer + XGBClassifier(hist, 500 trees, balanced sample weights)
- `mlp`: median imputer + StandardScaler + MLPClassifier(128 hidden units, max_iter=30)
- `lgbm`: median imputer + LightGBM balanced, existing strong holdout params, no early stopping

## OOF RESULTS

| Model | OOF F1 @ argmax | Best OOF F1 | Best threshold | Accuracy | Class 0 Recall | Class 1 Recall | Positive % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lgbm | 0.800896 | N/A | N/A | 0.780003 | 0.918424 | 0.870187 | N/A |
| mlp | 0.796818 | N/A | N/A | 0.785629 | 0.910653 | 0.868599 | N/A |
| lr | 0.794967 | N/A | N/A | 0.788027 | 0.912623 | 0.860659 | N/A |
| xgb | 0.786742 | N/A | N/A | 0.765391 | 0.914885 | 0.886066 | N/A |

Threshold note: this is a 7-class single-label task, so binary threshold and positive percentage are not applicable. Predictions use probability argmax.

## Probability Correlations And Prediction Disagreement

| Model A | Model B | Probability Corr | Agreement | Disagreement | A Correct/B Wrong | B Correct/A Wrong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| lr | xgb | 0.951195 | 0.863510 | 0.136490 | 15626 | 10464 |
| lr | mlp | 0.943227 | 0.857301 | 0.142699 | 13954 | 13407 |
| lr | lgbm | 0.970403 | 0.886550 | 0.113450 | 11828 | 9998 |
| xgb | mlp | 0.911748 | 0.821557 | 0.178443 | 14822 | 19437 |
| xgb | lgbm | 0.976526 | 0.904398 | 0.095602 | 7387 | 10719 |
| mlp | lgbm | 0.934502 | 0.841277 | 0.158723 | 15885 | 14602 |

## Simple Blend Results

| Blend | Macro F1 | Weights |
| --- | ---: | --- |
| equal_weight_useful | 0.814605 | `{'lgbm': 0.25, 'mlp': 0.25, 'lr': 0.25, 'xgb': 0.25}` |
| lgbm_mlp_50_50 | 0.813958 | `{'lgbm': 0.5, 'mlp': 0.5}` |
| lgbm_mlp_70_30 | 0.813399 | `{'lgbm': 0.7, 'mlp': 0.3}` |
| lgbm_mlp_80_20 | 0.809869 | `{'lgbm': 0.8, 'mlp': 0.2}` |
| lgbm_alone | 0.800896 | `{'lgbm': 1.0}` |

## BEST OOF CANDIDATE

Model: `lgbm`
Macro F1: `0.800896`
Threshold: `N/A for multiclass argmax`

## BLEND RESULTS

Best simple blend: `equal_weight_useful`
Macro F1: `0.814605`
Weights: `{'lgbm': 0.25, 'mlp': 0.25, 'lr': 0.25, 'xgb': 0.25}`

## Comparison Against Incumbent

The incumbent is a leaderboard score (`0.81534`) with unknown local reproduction. This OOF benchmark is for model comparison only; it does not prove leaderboard improvement.