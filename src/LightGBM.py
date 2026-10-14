"""
LightGBM.py
Milestone 3: Fast Histogram-Based Gradient Boosting for VitalSign (50,000 Records)
Uses HistGradientBoostingClassifier / LightGBM for rapid inference and non-linear risk estimation.
"""

import os
import json
import pandas as pd
import numpy as np

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGBM = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'LightGBM_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_lightgbm():
    print("=" * 70)
    print("  MILESTONE 3: LIGHTGBM / HIST-GRADIENT BOOSTING (50,000 RECORDS)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(50000)

    if HAS_LGBM:
        print("[+] Training native LightGBM Classifier...")
        model = LGBMClassifier(n_estimators=100, learning_rate=0.08, random_state=42, verbose=-1)
    else:
        print("[+] Training Scikit-Learn HistGradientBoosting Classifier...")
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)

    print("\n--- LIGHTGBM BENCHMARK SCORES ---")
    print(f"Accuracy  : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | AUC: {auc:.2f}%")

    metrics_dict = {
        'model_name': 'LightGBM / HistGradientBoosting',
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2)
    }
    with open(os.path.join(OUTPUT_DIR, 'lightgbm_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    pd.DataFrame([metrics_dict]).to_csv(os.path.join(OUTPUT_DIR, 'lightgbm_metrics.csv'), index=False)
    print(f"[[OK]] Exported LightGBM outputs to {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_dict

if __name__ == '__main__':
    run_lightgbm()

