"""
logistic_regression_binary_classify.py
Milestone 3: Logistic Regression Binary Classifier for 30-Day Readmission
Outputs metrics, confusion matrix, ROC curve, and prediction CSVs into output/Logistic_Regression_Outputs/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Logistic_Regression_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_logistic_regression():
    print("=" * 70)
    print("  MILESTONE 3: LOGISTIC REGRESSION BINARY CLASSIFIER (50k DATASET)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, feature_cols = load_and_preprocess_data(50000)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- LOGISTIC REGRESSION PERFORMANCE SCORES ---")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall    : {rec:.2f}%")
    print(f"F1-Score  : {f1:.2f}%")
    print(f"ROC-AUC   : {auc:.2f}%")

    # Save metrics JSON & Classification Report
    metrics_data = {
        'model': 'Logistic Regression',
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2),
        'confusion_matrix': cm
    }
    with open(os.path.join(OUTPUT_DIR, 'logistic_regression_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stable (0)', 'Readmit (1)'],
                yticklabels=['Stable (0)', 'Readmit (1)'])
    plt.title("Logistic Regression - Confusion Matrix", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save Predictions CSV
    pred_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred, 'Probability': np.round(y_prob, 4)})
    pred_df.head(200).to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

    print(f"[[OK]] Logistic Regression artifacts generated successfully in {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_data

if __name__ == '__main__':
    run_logistic_regression()

