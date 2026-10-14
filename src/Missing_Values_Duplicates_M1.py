"""
Missing_Values_Duplicates_M1.py
Milestone 1: In-depth Missing Values and Duplicates Audit for VitalSign 50k Dataset
Outputs statistical tables and reports to output/EDA_Analysis_outputs/
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'EDA_Analysis_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def audit_missing_and_duplicates():
    print("=" * 70)
    print("  MILESTONE 1: MISSING VALUES & DUPLICATES AUDIT (50k RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    total_rows = len(df)

    # Missing values analysis
    null_counts = df.isnull().sum()
    null_pct = (null_counts / total_rows * 100).round(3)
    data_types = df.dtypes

    audit_df = pd.DataFrame({
        'Feature': df.columns,
        'Data_Type': [str(t) for t in data_types],
        'Missing_Count': null_counts.values,
        'Missing_Percentage': [f"{p}%" for p in null_pct.values],
        'Unique_Values': [df[c].nunique() for c in df.columns]
    })

    print(audit_df.to_string(index=False))

    dup_count = int(df.duplicated().sum())
    print(f"\n[+] Total Duplicates Detected: {dup_count}")

    # Export audit CSV
    csv_out = os.path.join(OUTPUT_DIR, 'missing_and_duplicates_summary.csv')
    audit_df.to_csv(csv_out, index=False)
    print(f"[[OK]] Exported Missing Values & Duplicates Summary to: {csv_out}")
    print("=" * 70)

if __name__ == '__main__':
    audit_missing_and_duplicates()

