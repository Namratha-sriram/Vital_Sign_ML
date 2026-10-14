"""
Correlation_Matrix_heatmap_boxplots_M1.py
Milestone 1: Correlation Matrix, Heatmap & Outlier Boxplots for VitalSign 50k Dataset
Generates correlation heatmaps and vital signs distribution boxplots.
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

def generate_correlation_and_boxplots():
    print("=" * 70)
    print("  MILESTONE 1: CORRELATION MATRIX & BOXPLOTS (50k VITALSIGN DATA)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)

    # 1. Numerical Correlation Matrix
    numeric_cols = [
        'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'blood_glucose', 'respiratory_rate', 'body_temperature',
        'oxygen_saturation', 'time_in_hospital', 'num_lab_procedures',
        'num_medications', 'number_emergency', 'readmitted'
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[numeric_cols].corr()

    # Save correlation heatmap
    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='coolwarm',
        mask=mask, cbar_kws={'label': 'Pearson Correlation'},
        linewidths=0.5
    )
    plt.title("VitalSign 50k - Clinical Features Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_img = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    plt.savefig(corr_img, dpi=150)
    plt.close()
    print(f"[[OK]] Saved Correlation Heatmap to: {corr_img}")

    # 2. Vitals Boxplots by Readmission Status
    vitals = ['heart_rate', 'systolic_bp', 'blood_glucose', 'oxygen_saturation']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    colors = ['#2f7855', '#b23a2e']
    labels = ['Stable (0)', 'Readmitted (1)']

    for i, col in enumerate(vitals):
        sns.boxplot(
            data=df, x='readmitted', y=col, ax=axes[i],
            palette=colors, width=0.45
        )
        axes[i].set_title(f"{col.replace('_', ' ').title()} vs Readmission", fontsize=11, fontweight='bold')
        axes[i].set_xticklabels(labels)
        axes[i].set_xlabel("")

    plt.suptitle("Vital Signs Distribution Across Patient Outcomes (50,000 Records)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    box_img = os.path.join(OUTPUT_DIR, 'vitals_outcome_boxplots.png')
    plt.savefig(box_img, dpi=150)
    plt.close()
    print(f"[[OK]] Saved Vitals Boxplots to: {box_img}")
    print("=" * 70)

if __name__ == '__main__':
    generate_correlation_and_boxplots()

