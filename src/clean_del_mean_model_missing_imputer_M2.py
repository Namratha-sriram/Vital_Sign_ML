"""
clean_del_mean_model_missing_imputer_M2.py
Milestone 2: Missing Data Treatment & Imputation Strategies (Mean, Median, Mode & Model)
Demonstrates handling of incomplete physiological observations across 50,000 records.
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_imputation_methods():
    print("=" * 70)
    print("  MILESTONE 2: MISSING VALUE IMPUTATION STRATEGIES (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    
    # Simulate a realistic test mask of 5% missing values in vital signs for evaluation
    vitals = ['blood_glucose', 'systolic_bp', 'heart_rate', 'body_temperature']
    df_missing = df.copy()
    np.random.seed(42)
    for v in vitals:
        mask = np.random.rand(len(df_missing)) < 0.05
        df_missing.loc[mask, v] = np.nan

    print("\n--- SYNTHETIC MISSINGNESS INJECTED FOR EXPERIMENTATION ---")
    print(df_missing[vitals].isnull().sum())

    # Strategy 1: Mean Imputation
    mean_imputer = SimpleImputer(strategy='mean')
    df_mean = df_missing[vitals].copy()
    df_mean[:] = mean_imputer.fit_transform(df_mean)

    # Strategy 2: Median Imputation
    median_imputer = SimpleImputer(strategy='median')
    df_median = df_missing[vitals].copy()
    df_median[:] = median_imputer.fit_transform(df_median)

    # Strategy 3: Listwise Deletion (Dropna)
    df_dropped = df_missing.dropna(subset=vitals)

    print("\n--- IMPUTATION STRATEGY COMPARISON (BLOOD GLUCOSE) ---")
    print(f"Original Ground Truth Mean : {df['blood_glucose'].mean():.2f} (std: {df['blood_glucose'].std():.2f})")
    print(f"Mean Imputed Mean          : {df_mean['blood_glucose'].mean():.2f} (std: {df_mean['blood_glucose'].std():.2f})")
    print(f"Median Imputed Mean        : {df_median['blood_glucose'].mean():.2f} (std: {df_median['blood_glucose'].std():.2f})")
    print(f"Listwise Dropped Mean      : {df_dropped['blood_glucose'].mean():.2f} (Rows retained: {len(df_dropped):,})")

    # Export cleaned imputed sample
    out_file = os.path.join(OUTPUT_DIR, '02_imputed_vitals_sample.csv')
    df_mean.head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported imputed vitals sample to: {out_file}")
    print("=" * 70)

if __name__ == '__main__':
    compare_imputation_methods()

