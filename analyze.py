import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

bench_df = pd.read_csv('submission (1).csv')
ens_df = pd.read_csv('submission_ensemble.csv')
lgb_df = pd.read_csv('submission_lgb.csv')
lr_df = pd.read_csv('submission_lr.csv')

print("Phase 1: Benchmark Class Distribution")
bench_counts = bench_df['target'].value_counts().sort_index()
print(bench_counts)
print(bench_counts / len(bench_df))
print("\n")

print("Phase 3: ID-by-ID Comparison")
sources = {
    'Ensemble (80.456%)': ens_df,
    'LightGBM': lgb_df,
    'Logistic Reg': lr_df
}

table = []
for name, df in sources.items():
    same = (bench_df['target'] == df['target']).sum()
    diff = len(bench_df) - same
    pct = same / len(bench_df) * 100
    table.append({'Source': name, 'Same as 81.162%': same, 'Different': diff, 'Agreement %': pct})

comp_df = pd.DataFrame(table)
print(comp_df.to_markdown(index=False))
print("\n")

print("Confusion Matrix with Ensemble")
cm = confusion_matrix(ens_df['target'], bench_df['target'])
print("Rows: Ensemble, Cols: Benchmark (81.162%)")
print(cm)

print("\nClass distribution differences (Ensemble vs Benchmark):")
ens_counts = ens_df['target'].value_counts().sort_index()
diff = bench_counts - ens_counts
print(diff)
