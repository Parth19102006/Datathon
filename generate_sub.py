import numpy as np
import pandas as pd
import json
import os

print('Loading files...')
lgbm_test = np.load('lgbm_test_proba.npy')
xgb_test = np.load('xgb_test_proba.npy')
cb_test = np.load('cb_test_proba.npy')

with open('class_order.json', 'r') as f:
    class_order = json.load(f)

print('Blending...')
p = 0.785 * lgbm_test + 0.145 * xgb_test + 0.070 * cb_test

multipliers = {
    1: 0.9738,
    2: 0.9612,
    3: 1.0244,
    4: 1.0075,
    5: 0.9891,
    6: 1.0257,
    7: 1.0419
}

print('Applying multipliers...')
for i, cls in enumerate(class_order):
    p[:, i] *= multipliers[cls]

print('Getting predictions...')
preds_idx = np.argmax(p, axis=1)
preds = [class_order[idx] for idx in preds_idx]

print('Creating submission...')
test_df = pd.read_csv('test.csv', usecols=['id'])

print(f'Rows matches test: {len(preds) == len(test_df)}')
print(f'Length of predictions: {len(preds)}')
print(f'Length of test_df: {len(test_df)}')
print(f'No NaNs in predictions: {pd.Series(preds).isna().sum() == 0}')
print(f'Valid labels: {set(preds).issubset(set(class_order))}')
print(f'All predicted labels are valid: {set(preds)}')

submission = pd.DataFrame({
    'id': test_df['id'],
    'target': preds
})

output_file = 'submission_phase4b_multipliers.csv'
submission.to_csv(output_file, index=False)
out_path = os.path.abspath(output_file)

print('--- VERIFICATION ---')
print(f'Exact file path: {out_path}')
print(f'Shape: {submission.shape}')
print(f'Columns: {list(submission.columns)}')

