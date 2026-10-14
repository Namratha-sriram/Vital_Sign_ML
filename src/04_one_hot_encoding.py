"""
04_one_hot_encoding.py
Milestone 2: One-Hot Dummy Encoding for VitalSign Clinical Features
Converts multi-class nominal features into discrete binary indicator variables.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_one_hot_encoding():
    print("=" * 70)
    print("  MILESTONE 2: ONE-HOT DUMMY ENCODING (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    target_cols = ['gender', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    print(f"\n[+] Ingested dataset: {len(df):,} records")
    print(f"[+] Categorical features for one-hot encoding: {target_cols}")

    df_dummies = pd.get_dummies(df[target_cols], columns=target_cols, drop_first=True, dtype=int)
    print(f"[+] Generated {df_dummies.shape[1]} binary indicators.")
    print("\n--- SAMPLE ENCODED DUMMY COLUMNS ---")
    print(df_dummies.iloc[:5, :6])

    out_file = os.path.join(OUTPUT_DIR, '04_one_hot_encoded_sample.csv')
    df_dummies.head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported One-Hot Encoded sample to: {out_file}")
    print("=" * 70)

if __name__ == '__main__':
    perform_one_hot_encoding()

