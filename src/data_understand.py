"""
data_understand.py
Milestone 1: Statistical Understanding of VitalSign Cohort (50,000 Records)
Generates descriptive statistics, demographic breakdowns, and risk profiles.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'EDA_Analysis_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def understand_cohort_data():
    print("=" * 70)
    print("  MILESTONE 1: CLINICAL COHORT UNDERSTANDING (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)

    total_patients = len(df)
    readm_count = int(df['readmitted'].sum())
    readm_pct = (readm_count / total_patients) * 100

    print(f"\n[+] Total Cohort Size: {total_patients:,} encounters")
    print(f"[+] Total Readmissions within 30 days: {readm_count:,} ({readm_pct:.2f}%)")

    # Demographic Breakdown
    print("\n--- GENDER DISTRIBUTION ---")
    print(df['gender'].value_counts(normalize=True).round(3) * 100)

    # Key Vitals Overview
    vitals = ['blood_glucose', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'oxygen_saturation', 'body_temperature']
    vitals_desc = df[vitals].describe().round(2).T
    print("\n--- PHYSIOLOGICAL VITALS STATISTICAL PROFILE ---")
    print(vitals_desc[['mean', 'std', 'min', '50%', 'max']])

    # Export statistical profile
    out_csv = os.path.join(OUTPUT_DIR, 'cohort_vitals_profile.csv')
    vitals_desc.to_csv(out_csv)
    print(f"\n[[OK]] Exported Cohort Profile to: {out_csv}")
    print("=" * 70)

if __name__ == '__main__':
    understand_cohort_data()

