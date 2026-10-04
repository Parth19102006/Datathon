import pandas as pd
import numpy as np

bench_df = pd.read_csv('submission (1).csv')
test_df = pd.read_csv('test.csv')

def proba_to_sub(proba_file):
    try:
        proba = np.load(proba_file)
        preds = np.argmax(proba, axis=1) + 1
        return preds
    except:
        return None

files = ['best_3way_test_proba.npy', 'cb_test_proba.npy', 'ensemble_test_proba.npy', 'lgbm_test_proba.npy', 'xgb_test_proba.npy', 'phase4b_best_3way_test_proba.npy']

for f in files:
    preds = proba_to_sub(f)
    if preds is not None:
        same = (bench_df['target'] == preds).sum()
        pct = same / len(bench_df) * 100
        print(f"{f}: {same} / {len(bench_df)} ({pct:.3f}%)")

