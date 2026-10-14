"""
03_embedding_encoding.py
Milestone 2: Categorical Frequency & Entity Embedding Representation
Maps categorical hospital admission and discharge codes into compact continuous embedding representations.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_embedding_encoding():
    print("=" * 70)
    print("  MILESTONE 2: ENTITY & FREQUENCY EMBEDDING ENCODING")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    target_cols = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']

    print(f"\n[+] Processing {len(df):,} patient records across nominal features: {target_cols}")

    df_embed = df.copy()
    for col in target_cols:
        # Frequency embedding representation
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df_embed[col + '_freq_embed'] = df[col].map(freq_map).round(4)
        
        # Mean target-based embedding (readmission probability embedding)
        target_map = df.groupby(col)['readmitted'].mean().to_dict()
        df_embed[col + '_risk_embed'] = df[col].map(target_map).round(4)

    embed_cols = [c for c in df_embed.columns if '_embed' in c]
    print("\n--- SAMPLE ENCOUNTER EMBEDDING MATRIX (FIRST 5 PATIENTS) ---")
    print(df_embed[target_cols + embed_cols].head(5))

    out_file = os.path.join(OUTPUT_DIR, '03_embedding_encoded_sample.csv')
    df_embed[target_cols + embed_cols].head(100).to_csv(out_file, index=False)
    print(f"\n[[OK]] Exported clinical embedding sample to: {out_file}")
    print("=" * 70)

if __name__ == '__main__':
    perform_embedding_encoding()

