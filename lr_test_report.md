# Logistic Regression Test Submission Report

True test Macro F1 cannot be calculated because `test.csv` does not contain hidden target labels.
**Do not interpret any figure below as a test F1 score.**

The selected model was already validated on the Phase 3.5 hold-out split
(Validation Macro F1 = 0.7959 / 79.59%).

## Data

- Train shape: 228039 rows × 176 columns
- Test shape: 97731 rows × 175 columns
- ID column: `id`
- Target column (train only): `target`
- Number of features: 174
- Feature columns: `f1` … `f174`
- Test feature columns exactly match train feature columns: True
- Test missing-value count: 2550057
- Train missing-value count: 5949963
- Valid target classes: [1, 2, 3, 4, 5, 6, 7]

## Best LR configuration (from Phase 3.5 results)

Source: `phase3_5_results.csv` (Logistic Regression trial with Macro F1 = 0.795856).

- C: 10.0
- solver: lbfgs
- penalty: l2
- class_weight: None
- max_iter: 1000
- random_state: 42
- n_jobs: -1

## Preprocessing configuration

- Imputation strategy: median (`SimpleImputer`)
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

- Training row count: 228039
- Test row count: 97731
- Number of predictions: 97731

## Predicted class distribution

- class 1: 11637 (11.9072%)
- class 2: 1064 (1.0887%)
- class 3: 23445 (23.9893%)
- class 4: 22549 (23.0725%)
- class 5: 11214 (11.4744%)
- class 6: 27487 (28.1252%)
- class 7: 335 (0.3428%)

## Submission

- File: `C:\Users\ASUS\OneDrive\Desktop\Datathon\submission_lr.csv`
- Submission shape: 97731 rows × 2 columns
- Submission columns: ['id', 'target']
- sample_submission.csv used: False

## Validation checks

- correct_n_rows: True
- correct_n_columns: True
- correct_column_names: True
- correct_column_order: True
- no_missing_values: True
- ids_match_test_csv: True
- ids_preserve_original_order: True
- prediction_classes_valid: True
- n_predictions_equals_n_test: True
- no_missing_predictions: True
- row_order_preserved: True

## First 10 rows

```
     id  target
1094682       2
1150260       3
1196783       3
1257567       4
1176766       4
1063479       3
1101996       2
1140779       3
1151597       1
1067794       3
```

## Last 10 rows

```
     id  target
1325097       4
1035223       1
1270847       1
1166907       4
1324579       6
1143145       3
1241895       4
1232757       6
1073932       1
1191967       4
```

## Final status

**submission_lr.csv is ready for Kaggle submission.**
