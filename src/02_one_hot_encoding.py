"""
02_one_hot_encoding.py
Module 2: One-Hot Dummy Encoding for VitalSign Dataset
Demonstrates one-hot binary encoding on nominal multi-category features:
- admission_type_id (Emergency, Urgent, Elective, Newborn, etc.)
- admission_source_id (Physician Referral, Clinic, Emergency Room, etc.)
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_one_hot_encoding():
    print("=" * 65)
    print("  VITALSIGN ML PIPELINE - 02 ONE-HOT DUMMY ENCODING")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"\n[+] Ingested dataset: {len(df):,} records, {df.shape[1]} columns")

    target_cols = ['admission_type_id', 'admission_source_id']
    print(f"[+] Nominal category features selected for dummy encoding: {target_cols}")

    print("\n--- SAMPLE BEFORE ONE-HOT ENCODING ---")
    print(df[target_cols].head(5))

    # Convert numeric IDs to categorical strings for clear dummy column names
    df_cat = df[target_cols].copy()
    df_cat['admission_type_id'] = 'adm_type_' + df_cat['admission_type_id'].astype(str)
    df_cat['admission_source_id'] = 'adm_src_' + df_cat['admission_source_id'].astype(str)

    # Perform One-Hot Encoding via pandas get_dummies
    df_dummies = pd.get_dummies(df_cat, prefix=['type', 'source'], dtype=int)

    print(f"\n[+] Generated {df_dummies.shape[1]} binary dummy columns:")
    for col in df_dummies.columns[:10]:
        print(f"    - {col}")
    if df_dummies.shape[1] > 10:
        print(f"    ... and {df_dummies.shape[1] - 10} more columns")

    print("\n--- SAMPLE AFTER ONE-HOT ENCODING (First 5 Dummy Columns) ---")
    print(df_dummies.iloc[:5, :5])

    # Export sample
    out_file = os.path.join(OUTPUT_DIR, '02_one_hot_encoded_sample.csv')
    df_dummies.head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported one-hot encoded sample (100 rows) to: {out_file}")
    print("=" * 65)

if __name__ == "__main__":
    perform_one_hot_encoding()

