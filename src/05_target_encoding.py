"""
05_target_encoding.py
Milestone 2: Supervised Target Encoding with Smoothing for VitalSign Dataset
Replaces categorical levels with regularized mean posterior probability of 30-day readmission.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_target_encoding(smoothing=10):
    print("=" * 70)
    print("  MILESTONE 2: SUPERVISED TARGET ENCODING (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    target_cols = ['admission_type_id', 'admission_source_id', 'discharge_disposition_id']
    global_mean = df['readmitted'].mean()

    print(f"\n[+] Global Readmission Rate: {global_mean:.4f}")
    df_te = df.copy()

    for col in target_cols:
        # Smoothed empirical Bayes target encoding formula:
        # S_i = (count_i * mean_i + smoothing * global_mean) / (count_i + smoothing)
        stats = df.groupby(col)['readmitted'].agg(['count', 'mean'])
        smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        df_te[col + '_target_encoded'] = df[col].map(smoothed).round(4)

    te_cols = [c + '_target_encoded' for c in target_cols]
    print("\n--- SAMPLE TARGET ENCODED FEATURES ---")
    print(df_te[target_cols + te_cols].head(5))

    out_file = os.path.join(OUTPUT_DIR, '05_target_encoded_sample.csv')
    df_te[target_cols + te_cols].head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported Target Encoded sample to: {out_file}")
    print("=" * 70)

if __name__ == '__main__':
    perform_target_encoding()

