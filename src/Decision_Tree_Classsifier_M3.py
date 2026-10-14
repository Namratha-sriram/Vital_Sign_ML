"""
Decision_Tree_Classsifier_M3.py
Milestone 3: Decision Tree Classifier for VitalSign Readmission Risk
Evaluates interpretability, tree depth, confusion matrix, and feature importances.
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

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Decision_Tree_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_decision_tree():
    print("=" * 70)
    print("  MILESTONE 3: DECISION TREE CLASSIFIER (50,000 RECORDS)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, feature_cols = load_and_preprocess_data(50000)

    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n--- DECISION TREE SCORES ---")
    print(f"Accuracy  : {acc:.2f}% | F1: {f1:.2f}% | AUC: {auc:.2f}%")

    metrics_data = {
        'model': 'Decision Tree',
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2),
        'confusion_matrix': cm
    }
    with open(os.path.join(OUTPUT_DIR, 'decision_tree_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # Export Top Feature Importances
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fi_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Stable (0)', 'Readmit (1)'],
                yticklabels=['Stable (0)', 'Readmit (1)'])
    plt.title("Decision Tree - Confusion Matrix", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Plot pruned Tree Architecture
    plt.figure(figsize=(16, 8))
    plot_tree(model, max_depth=3, feature_names=feature_cols, class_names=['Stable', 'Readmit'], filled=True, rounded=True, fontsize=8)
    plt.title("Decision Tree Decision Hierarchy (Top 3 Levels)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tree_structure.png'), dpi=150)
    plt.close()

    print(f"[[OK]] Decision Tree exported to {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_data

if __name__ == '__main__':
    run_decision_tree()

