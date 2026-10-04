import pandas as pd
import numpy as np
import time
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def main():
    print("Loading data...")
    df = pd.read_csv('train.csv')

    id_col = 'id'
    target_col = 'target'
    
    # We will exclude ID column
    X = df.drop(columns=[id_col, target_col])
    y = df[target_col]

    # Encode labels to 0-6 to be safe for all models (XGBoost needs this, etc.)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/validation split (80/20)
    print("Splitting data (Stratified 80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    print("Target class distribution in train:", pd.Series(y_train).value_counts(normalize=True).to_dict())

    # Define models
    models = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42))
        ]),
        'Extra Trees': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42))
        ]),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100, random_state=42, class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, n_jobs=-1, random_state=42, objective='multi:softmax', num_class=7
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, n_jobs=-1, random_state=42, class_weight='balanced'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, thread_count=-1, random_state=42, verbose=False, auto_class_weights='Balanced'
        )
    }

    results = []
    reports = {}
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            start_pred = time.time()
            y_pred = model.predict(X_val)
            pred_time = time.time() - start_pred
            
            acc = accuracy_score(y_val, y_pred)
            macro_f1 = f1_score(y_val, y_pred, average='macro')
            weighted_f1 = f1_score(y_val, y_pred, average='weighted')
            
            print(f"{name} - Acc: {acc:.4f}, Macro F1: {macro_f1:.4f}, Time: {train_time:.2f}s")
            
            results.append({
                'Model': name,
                'Accuracy': acc,
                'Macro F1': macro_f1,
                'Weighted F1': weighted_f1,
                'Training Time': train_time,
                'Prediction Time': pred_time
            })
            
            reports[name] = classification_report(y_val, y_pred, output_dict=True)
            confusion_matrices[name] = confusion_matrix(y_val, y_pred).tolist()
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results.append({
                'Model': name,
                'Accuracy': None,
                'Macro F1': None,
                'Weighted F1': None,
                'Training Time': None,
                'Prediction Time': None,
                'Error': str(e)
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('phase3_results.csv', index=False)
    
    with open('phase3_reports.json', 'w') as f:
        json.dump({
            'classes': le.classes_.tolist(),
            'classification_reports': reports,
            'confusion_matrices': confusion_matrices
        }, f, indent=4)
        
    print("\nPhase 3 Baseline Benchmarking Complete.")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
