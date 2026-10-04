import pandas as pd
import numpy as np
import json
import itertools
from collections import defaultdict
from scipy.stats import skew, entropy, chi2_contingency

print("Loading data...")
df = pd.read_csv('train.csv')

features = [f'f{i}' for i in range(1, 175)]
id_col = 'id'
target_col = 'target'

results = {}

# 1. Unique-value analysis
print("1. Unique-value analysis...")
unique_stats = {}
constant_features = []
near_constant_features = []
low_cardinality = []

for f in features:
    non_null = df[f].notna().sum()
    nunique = df[f].nunique()
    pct_unique = nunique / non_null if non_null > 0 else 0
    unique_stats[f] = {'nunique': int(nunique), 'non_null': int(non_null), 'pct_unique': float(pct_unique)}
    if nunique <= 1:
        constant_features.append(f)
    elif nunique <= 10:
        near_constant_features.append(f)
    elif pct_unique < 0.01:
        low_cardinality.append(f)

results['unique'] = {
    'constant': constant_features,
    'near_constant': near_constant_features,
    'low_cardinality_count': len(low_cardinality)
}

# 2. Statistical summary
print("2. Statistical summary...")
stat_summary = {}
high_skew = []
large_range = []
high_var = []

# pre-calculate for performance
desc = df[features].describe(percentiles=[.25, .5, .75]).T
skews = df[features].skew()

for f in features:
    q1 = desc.loc[f, '25%']
    q3 = desc.loc[f, '75%']
    iqr = q3 - q1
    s = skews[f]
    rng = desc.loc[f, 'max'] - desc.loc[f, 'min']
    stat_summary[f] = {
        'mean': desc.loc[f, 'mean'],
        'median': desc.loc[f, '50%'],
        'std': desc.loc[f, 'std'],
        'min': desc.loc[f, 'min'],
        'max': desc.loc[f, 'max'],
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skew': s
    }
    if abs(s) > 2:
        high_skew.append(f)
    if rng > 100000: # heuristic
        large_range.append(f)
    if desc.loc[f, 'std'] > 10000: # heuristic
        high_var.append(f)

results['stats'] = {
    'high_skew_count': len(high_skew),
    'large_range_count': len(large_range),
    'high_var_count': len(high_var),
    'high_skew_examples': high_skew[:5]
}

# 3. Outlier analysis
print("3. Outlier analysis...")
outlier_stats = {}
for f in features:
    q1 = stat_summary[f]['q1']
    q3 = stat_summary[f]['q3']
    iqr = stat_summary[f]['iqr']
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    non_null_mask = df[f].notna()
    if non_null_mask.sum() == 0 or iqr == 0:
        outlier_stats[f] = 0
        continue
        
    outliers = ((df[f] < lower_bound) | (df[f] > upper_bound)) & non_null_mask
    outlier_stats[f] = outliers.sum() / non_null_mask.sum()

top_outliers = sorted(outlier_stats.items(), key=lambda x: x[1], reverse=True)[:20]
results['outliers'] = {
    'top_20': [(f, float(v)) for f, v in top_outliers]
}

# 4. Distribution analysis
print("4. Distribution analysis...")
approx_sym = [f for f in features if abs(stat_summary[f]['skew']) < 0.5][:5]
high_skew_dist = [f for f in features if abs(stat_summary[f]['skew']) > 2][:5]
high_outlier_dist = [f for f, _ in top_outliers][:5]
results['distributions'] = {
    'approx_symmetric': approx_sym,
    'highly_skewed': high_skew_dist,
    'high_outliers': high_outlier_dist
}

# 5. Correlation analysis
print("5. Correlation analysis...")
# Sample for faster correlation if needed, but 200k rows is fine
corr_matrix = df[features].corr().abs().values
# Extract upper triangle
upper_tri = np.triu(corr_matrix, k=1)

corr_90 = np.sum(upper_tri > 0.90)
corr_95 = np.sum(upper_tri > 0.95)
corr_99 = np.sum(upper_tri > 0.99)

# Find top correlations
indices = np.unravel_index(np.argsort(upper_tri, axis=None)[::-1], upper_tri.shape)
top_corrs = []
for i in range(10): # top 10
    idx1, idx2 = indices[0][i], indices[1][i]
    if upper_tri[idx1, idx2] > 0:
        top_corrs.append((features[idx1], features[idx2], float(upper_tri[idx1, idx2])))

results['correlations'] = {
    'gt_90': int(corr_90),
    'gt_95': int(corr_95),
    'gt_99': int(corr_99),
    'top_10': top_corrs
}

# 6. Missingness relationship
print("6. Missingness relationship...")
missing_df = df[features].isna().astype(int)
miss_corr = missing_df.corr().abs().values
miss_upper_tri = np.triu(miss_corr, k=1)

miss_indices = np.unravel_index(np.argsort(miss_upper_tri, axis=None)[::-1], miss_upper_tri.shape)
top_miss_corrs = []
for i in range(10): # top 10
    idx1, idx2 = miss_indices[0][i], miss_indices[1][i]
    if miss_upper_tri[idx1, idx2] > 0:
        top_miss_corrs.append((features[idx1], features[idx2], float(miss_upper_tri[idx1, idx2])))

results['missing_correlations'] = {
    'top_10': top_miss_corrs
}

# 7. Missingness indicators vs target
print("7. Missingness indicators vs target...")
# We use max absolute difference in target class proportions
target_counts = df[target_col].value_counts(normalize=True).sort_index()
miss_vs_target = {}
for f in features:
    # Target distribution when missing
    miss_mask = df[f].isna()
    if miss_mask.sum() == 0 or (~miss_mask).sum() == 0:
        miss_vs_target[f] = 0
        continue
        
    dist_miss = df.loc[miss_mask, target_col].value_counts(normalize=True).sort_index()
    dist_obs = df.loc[~miss_mask, target_col].value_counts(normalize=True).sort_index()
    
    # Align indices
    all_classes = target_counts.index
    dist_miss = dist_miss.reindex(all_classes).fillna(0)
    dist_obs = dist_obs.reindex(all_classes).fillna(0)
    
    max_diff = (dist_miss - dist_obs).abs().max()
    miss_vs_target[f] = max_diff

top_miss_target = sorted(miss_vs_target.items(), key=lambda x: x[1], reverse=True)[:10]
results['missing_vs_target'] = {
    'top_10': [(f, float(v)) for f, v in top_miss_target]
}

# 8. ID analysis
print("8. ID analysis...")
id_series = df[id_col]
id_nunique = id_series.nunique()
id_len = len(id_series)
id_min = id_series.min()
id_max = id_series.max()
id_sequential = bool((id_max - id_min + 1) == id_len and id_nunique == id_len)

results['id_analysis'] = {
    'unique_count': int(id_nunique),
    'duplicate_count': int(id_len - id_nunique),
    'min': int(id_min),
    'max': int(id_max),
    'is_sequential': id_sequential
}

print("Saving results...")
with open('audit_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Done.")
