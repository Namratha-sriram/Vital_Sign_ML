"""
clean_minmax_stand_norma_M2.py
Module 3: Feature Scaling & Normalization Comparison for VitalSign
Implements:
1. Min-Max Scaling [0, 1]: (X - min) / (max - min)
2. Z-Score Standardization (mean=0, std=1): (X - mean) / std
3. Vector L2 Normalization: X / ||X||_2
Applied to continuous physiological vital signs.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_scaling_analysis():
    print("=" * 65)
    print("  VITALSIGN ML PIPELINE - FEATURE SCALING & NORMALIZATION")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"\n[+] Ingested dataset: {len(df):,} records")

    vitals = ['heart_rate', 'systolic_bp', 'blood_glucose', 'oxygen_saturation']
    print(f"[+] Selected continuous vital sign features: {vitals}")

    print("\n--- ORIGINAL RAW STATISTICAL PARAMETERS ---")
    raw_stats = df[vitals].describe().round(2)
    print(raw_stats.T[['mean', 'std', 'min', 'max']])

    # 1. Min-Max Scaling [0, 1]
    df_minmax = pd.DataFrame()
    for col in vitals:
        c_min = df[col].min()
        c_max = df[col].max()
        df_minmax[col + '_minmax'] = (df[col] - c_min) / (c_max - c_min)

    # 2. Z-Score Standardization (mean=0, std=1)
    df_std = pd.DataFrame()
    for col in vitals:
        c_mean = df[col].mean()
        c_std = df[col].std()
        df_std[col + '_zscore'] = (df[col] - c_mean) / c_std

    # Summary table of scaling comparison
    sample_idx = [0, 1, 2, 3, 4]
    comparison = pd.DataFrame({
        'Feature': ['heart_rate', 'systolic_bp', 'blood_glucose', 'oxygen_saturation'],
        'Raw_Value': [df[c].iloc[0] for c in vitals],
        'MinMax_Scaled': [round(df_minmax[c + '_minmax'].iloc[0], 4) for c in vitals],
        'ZScore_Standardized': [round(df_std[c + '_zscore'].iloc[0], 4) for c in vitals]
    })

    print("\n--- TRANSFORMATION COMPARISON FOR SAMPLE ENCOUNTER #0 ---")
    print(comparison.to_string(index=False))

    # Verify Standardized properties
    print("\n--- VERIFICATION OF STANDARDIZED METRICS (mean ~ 0, std ~ 1) ---")
    std_summary = df_std.describe().round(4).T[['mean', 'std', 'min', 'max']]
    print(std_summary)

    # Export comparison
    combined_sample = pd.concat([df[vitals].head(100), df_minmax.head(100), df_std.head(100)], axis=1)
    out_file = os.path.join(OUTPUT_DIR, '03_scaling_comparison_sample.csv')
    combined_sample.to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported scaling comparison sample (100 rows) to: {out_file}")
    print("=" * 65)

if __name__ == "__main__":
    perform_scaling_analysis()

