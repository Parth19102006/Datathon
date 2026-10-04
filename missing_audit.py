import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

# Create directories
output_dir = r"C:\Users\ASUS\.gemini\antigravity\brain\947decf9-c0b4-47f8-a5e7-33bb755d95cc\scratch\missing_audit"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = pd.read_csv(r'c:\Users\ASUS\OneDrive\Desktop\Datathon\train.csv')

# 2. Identify columns
id_col = 'id' if 'id' in df.columns else None
target_col = [c for c in df.columns if 'target' in c.lower() or 'class' in c.lower()][0]
feature_cols = [c for c in df.columns if c not in [id_col, target_col]]

print(f"Dataset shape: {df.shape}")
print(f"ID Column: {id_col}")
print(f"Target Column: {target_col}")
print(f"Feature Columns: {len(feature_cols)}\n")

# 3. Calculate overall missingness
total_cells = np.prod(df[feature_cols].shape)
total_missing = df[feature_cols].isnull().sum().sum()
cols_with_missing = (df[feature_cols].isnull().sum() > 0).sum()

print(f"Total missing values: {total_missing}")
print(f"Columns containing missing values: {cols_with_missing}")

# 4. Feature level missingness
missing_counts = df[feature_cols].isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
non_missing_counts = len(df) - missing_counts

missing_df = pd.DataFrame({
    'Feature': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': missing_pct.values,
    'Non_Missing_Count': non_missing_counts.values
}).sort_values('Missing_Percentage', ascending=False)

print("\nTop 20 Features by Missing Percentage:")
print(missing_df.head(20).to_string(index=False))

# 5 & 6. Group features into categories
cat_0 = missing_df[missing_df['Missing_Percentage'] == 0]
cat_0_5 = missing_df[(missing_df['Missing_Percentage'] > 0) & (missing_df['Missing_Percentage'] <= 5)]
cat_5_20 = missing_df[(missing_df['Missing_Percentage'] > 5) & (missing_df['Missing_Percentage'] <= 20)]
cat_20_50 = missing_df[(missing_df['Missing_Percentage'] > 20) & (missing_df['Missing_Percentage'] <= 50)]
cat_50_80 = missing_df[(missing_df['Missing_Percentage'] > 50) & (missing_df['Missing_Percentage'] <= 80)]
cat_80 = missing_df[missing_df['Missing_Percentage'] > 80]

print("\nMissingness Categories:")
print(f"0% missing: {len(cat_0)} features")
print(f">0% and <=5%: {len(cat_0_5)} features")
print(f">5% and <=20%: {len(cat_5_20)} features")
print(f">20% and <=50%: {len(cat_20_50)} features")
print(f">50% and <=80%: {len(cat_50_80)} features")
print(f">80% missing: {len(cat_80)} features")

# 7. Identify features
print(f"\nFeatures with extremely high missingness (>80%): {cat_80['Feature'].tolist()}")
print(f"Features with moderate/high missingness (>50%): {pd.concat([cat_50_80, cat_80])['Feature'].tolist()}")
print(f"Features with relatively low missingness (<=5%): {pd.concat([cat_0, cat_0_5])['Feature'].tolist()}")

# 8. Missingness across 7 target classes for most-missing features (top 5)
top_5_missing = missing_df.head(5)['Feature'].tolist()
print("\nClass-wise missingness (%) for top 5 most-missing features:")
for f in top_5_missing:
    class_missing = df.groupby(target_col).apply(lambda x: x[f].isnull().mean() * 100)
    print(f"Feature {f}:")
    print(class_missing)

# 9. Row-wise missingness
row_missing = df[feature_cols].isnull().sum(axis=1)
row_missing_pct = (row_missing / len(feature_cols)) * 100

print("\nRow-wise missingness summary:")
print(f"Min: {row_missing.min()}")
print(f"Median: {row_missing.median()}")
print(f"Mean: {row_missing.mean():.2f}")
print(f"95th percentile: {np.percentile(row_missing, 95)}")
print(f"99th percentile: {np.percentile(row_missing, 99)}")
print(f"Max: {row_missing.max()}")

print(f"\nRows with >25% features missing: {(row_missing_pct > 25).sum()} ({(row_missing_pct > 25).mean()*100:.2f}%)")
print(f"Rows with >50% features missing: {(row_missing_pct > 50).sum()} ({(row_missing_pct > 50).mean()*100:.2f}%)")
print(f"Rows with >75% features missing: {(row_missing_pct > 75).sum()} ({(row_missing_pct > 75).mean()*100:.2f}%)")

# 10. Visualizations
plt.figure(figsize=(10, 6))
plt.hist(missing_df['Missing_Percentage'], bins=20, edgecolor='black')
plt.title('Distribution of Missing Percentage Across Features')
plt.xlabel('Missing Percentage (%)')
plt.ylabel('Number of Features')
plt.savefig(os.path.join(output_dir, 'feature_missing_dist.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(row_missing_pct, bins=30, edgecolor='black')
plt.title('Distribution of Missing Percentage Per Row')
plt.xlabel('Missing Percentage (%)')
plt.ylabel('Number of Rows')
plt.savefig(os.path.join(output_dir, 'row_missing_dist.png'))
plt.close()
print("Plots saved.")
