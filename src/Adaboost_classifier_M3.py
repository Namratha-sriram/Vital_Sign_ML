"""
Adaboost_classifier_M3.py
Milestone 3: AdaBoost Adaptive Boosting Classifier for VitalSign Dataset
Combines iterative decision stumps with adaptive sample weighting.
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'AdaBoost_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_adaboost():
    print("=" * 70)
    print("  MILESTONE 3: ADABOOST ENSEMBLE CLASSIFIER (50,000 RECORDS)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(50000)

    base_stump = DecisionTreeClassifier(max_depth=2, random_state=42)
    ada = AdaBoostClassifier(estimator=base_stump, n_estimators=60, learning_rate=0.8, random_state=42)
    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)
    y_prob = ada.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)

    print("\n--- ADABOOST BENCHMARK SCORES ---")
    print(f"Accuracy  : {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | AUC: {auc:.2f}%")

    metrics_dict = {
        'model_name': 'AdaBoost Classifier',
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2)
    }
    with open(os.path.join(OUTPUT_DIR, 'adaboost_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    pd.DataFrame([metrics_dict]).to_csv(os.path.join(OUTPUT_DIR, 'adaboost_metrics.csv'), index=False)
    print(f"[[OK]] Exported AdaBoost outputs to {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_dict

if __name__ == '__main__':
    run_adaboost()

