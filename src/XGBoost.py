"""
XGBoost.py
Milestone 3: Gradient Boosted Decision Trees (XGBoost / GradientBoosting)
Evaluates gradient boosted ensemble trees on 50,000 clinical records.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'XGBoost_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_xgboost():
    print("=" * 70)
    print("  MILESTONE 3: GRADIENT BOOSTING / XGBOOST CLASSIFIER (50k DATASET)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, feature_cols = load_and_preprocess_data(50000)

    if HAS_XGB:
        print("[+] Training native XGBoost Classifier...")
        model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    else:
        print("[+] Training Scikit-Learn Gradient Boosting Classifier...")
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- GRADIENT BOOSTING PERFORMANCE SCORES ---")
    print(f"Accuracy  : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | AUC: {auc:.2f}%")

    metrics_dict = {
        'model_name': 'XGBoost / Gradient Boosting',
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2),
        'confusion_matrix': cm
    }

    with open(os.path.join(OUTPUT_DIR, 'xgboost_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    pd.DataFrame([metrics_dict]).to_csv(os.path.join(OUTPUT_DIR, 'xgboost_metrics.csv'), index=False)
    print(f"[[OK]] Exported Gradient Boosting / XGBoost outputs to {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_dict

if __name__ == '__main__':
    run_xgboost()

