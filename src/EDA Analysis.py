"""
EDA Analysis.py
Milestone 1: Exploratory Data Analysis & Univariate Distribution Plots for VitalSign (50,000 Records)
Generates feature distribution charts, boxplots, and statistical profiling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'EDA_Analysis_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda_analysis():
    print("=" * 70)
    print("  MILESTONE 1: EXPLORATORY DATA ANALYSIS (EDA) - 50,000 RECORDS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"[+] Loaded {len(df):,} patient records")

    vitals = ['blood_glucose', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'oxygen_saturation']
    for v in vitals:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[v], kde=True, color='#35524c')
        plt.title(f"Univariate Distribution - {v.replace('_', ' ').title()}", fontsize=11, fontweight='bold')
        plt.xlabel(v)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{v}_distribution.png"), dpi=150)
        plt.close()

    print(f"[[OK]] Generated univariate distribution plots in {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == '__main__':
    run_eda_analysis()

