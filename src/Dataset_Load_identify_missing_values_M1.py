"""
Dataset_Load_identify_missing_values_M1.py
Milestone 1: Dataset Ingestion and Missing Value Identification
Analyzes the 50,000 VitalSign clinical records, scans for null/missing values,
checks row duplicates, and generates missingness visualizations.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'EDA_Analysis_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_inspect_dataset():
    print("=" * 70)
    print("  MILESTONE 1: VITALSIGN DATASET LOAD & MISSING VALUE IDENTIFICATION")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"\n[+] Successfully ingested: {len(df):,} patient records")
    print(f"[+] Total attributes / features: {df.shape[1]}")

    print("\n--- FIRST 5 CLINICAL RECORDS ---")
    print(df.head(5)[['patient_id', 'age', 'gender', 'heart_rate', 'systolic_bp', 'blood_glucose', 'readmitted']])

    print("\n--- DATA TYPES & SCHEMA OVERVIEW ---")
    print(df.dtypes)

    # Missing value identification
    print("\n--- MISSING VALUES PER ATTRIBUTE ---")
    missing_series = df.isnull().sum()
    print(missing_series)
    total_missing = missing_series.sum()
    print(f"\n[+] Total missing cells across 50,000 records: {total_missing}")

    # Duplicate detection
    duplicates = df.duplicated().sum()
    print(f"[+] Duplicate encounter records detected: {duplicates}")

    # Generate missingness heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title("VitalSign 50k Dataset - Missing Value Heatmap", fontsize=12, fontweight='bold')
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'missing_values_heatmap.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[[OK]] Saved Missing Values Heatmap to: {chart_path}")
    print("=" * 70)

if __name__ == '__main__':
    load_and_inspect_dataset()

