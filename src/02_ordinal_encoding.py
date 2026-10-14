"""
02_ordinal_encoding.py
Milestone 2: Categorical Ordinal Encoding for VitalSign Clinical Features
Preserves intrinsic medical severity orders for glucose, A1C, and insulin tiers.
"""

import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_ordinal_encoding():
    print("=" * 70)
    print("  MILESTONE 2: ORDINAL ENCODING FOR CLINICAL RISK FEATURES")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"\n[+] Loaded dataset: {len(df):,} records")

    # Ordinal clinical features with explicit clinical hierarchies
    categories_order = {
        'max_glu_serum': ['None', 'Norm', '>200', '>300'],
        'A1Cresult': ['None', 'Norm', '>7', '>8'],
        'insulin': ['No', 'Steady', 'Up', 'Down']
    }

    print("\n--- SAMPLE BEFORE ORDINAL ENCODING ---")
    print(df[list(categories_order.keys())].head(5))

    df_encoded = df.copy()
    for col, ordering in categories_order.items():
        oe = OrdinalEncoder(categories=[ordering], handle_unknown='use_encoded_value', unknown_value=-1)
        df_encoded[col + '_ordinal'] = oe.fit_transform(df_encoded[[col]])
        print(f"\nOrdinal Mapping for '{col}':")
        for idx, val in enumerate(ordering):
            print(f"  {val} -> {idx}")

    ordinal_cols = [c + '_ordinal' for c in categories_order.keys()]
    print("\n--- SAMPLE AFTER ORDINAL ENCODING ---")
    print(df_encoded[list(categories_order.keys()) + ordinal_cols].head(5))

    out_file = os.path.join(OUTPUT_DIR, '02_ordinal_encoded_sample.csv')
    df_encoded[list(categories_order.keys()) + ordinal_cols].head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported ordinal encoded sample to: {out_file}")
    print("=" * 70)

if __name__ == '__main__':
    perform_ordinal_encoding()

