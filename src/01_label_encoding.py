"""
01_label_encoding.py
Module 1: Categorical Label Encoding for VitalSign Dataset
Demonstrates label encoding on categorical variables:
- gender (Female: 0, Male: 1)
- max_glu_serum (None: 0, Norm: 1, >200: 2, >300: 3)
- A1Cresult (None: 0, Norm: 1, >7: 2, >8: 3)
- insulin (No: 0, Down: 1, Steady: 2, Up: 3)
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_label_encoding():
    print("=" * 65)
    print("  VITALSIGN ML PIPELINE - 01 LABEL ENCODING")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"\n[+] Ingested dataset: {len(df):,} records, {df.shape[1]} columns")

    cat_cols = ['gender', 'max_glu_serum', 'A1Cresult', 'insulin']
    print(f"[+] Target categorical columns for label encoding: {cat_cols}")

    # Display preview before encoding
    print("\n--- SAMPLE BEFORE ENCODING ---")
    print(df[cat_cols].head(5))

    # Define deterministic label mappings
    mappings = {
        'gender': {'Female': 0, 'Male': 1, 'Unknown/Invalid': 0},
        'max_glu_serum': {'None': 0, 'Norm': 1, '>200': 2, '>300': 3},
        'A1Cresult': {'None': 0, 'Norm': 1, '>7': 2, '>8': 3},
        'insulin': {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
    }

    df_encoded = df.copy()
    for col, mapping in mappings.items():
        if col in df_encoded.columns:
            df_encoded[col + '_encoded'] = df_encoded[col].map(mapping).fillna(0).astype(int)
            print(f"\nMapping for {col}:")
            for k, v in mapping.items():
                print(f"  '{k}' -> {v}")

    # Display preview after encoding
    encoded_cols = [c + '_encoded' for c in cat_cols if c in df.columns]
    print("\n--- SAMPLE AFTER LABEL ENCODING ---")
    print(df_encoded[cat_cols + encoded_cols].head(5))

    # Export output
    out_file = os.path.join(OUTPUT_DIR, '01_label_encoded_sample.csv')
    df_encoded[cat_cols + encoded_cols].head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported label-encoded sample (100 rows) to: {out_file}")
    print("=" * 65)

if __name__ == "__main__":
    perform_label_encoding()

