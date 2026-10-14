"""
Random_Forest_Tree_M3.py
Milestone 3: Production Random Forest Classifier for VitalSign Readmission Risk
Produces exhaustive outputs mirroring the capstone architecture:
1. Metrics, parameters, and classification reports in metrics/
2. High-res charts in charts/
3. Confusion matrix heatmap in confusion_matrix/
4. Clinical feature importances in feature_importance/
5. Serialized model in model/
6. Test predictions in predictions/
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Random_Forest_Tree_M3_Outputs')
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CM_DIR = os.path.join(OUTPUT_DIR, 'confusion_matrix')
FI_DIR = os.path.join(OUTPUT_DIR, 'feature_importance')
METRICS_DIR = os.path.join(OUTPUT_DIR, 'metrics')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
PRED_DIR = os.path.join(OUTPUT_DIR, 'predictions')

for d in [CHARTS_DIR, CM_DIR, FI_DIR, METRICS_DIR, MODEL_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

def run_random_forest():
    print("=" * 70)
    print("  MILESTONE 3: PRODUCTION RANDOM FOREST CLASSIFIER (50,000 RECORDS)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, _, feature_cols = load_and_preprocess_data(50000)

    # Optimal hyperparameters selected via grid search
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred) * 100)
    prec = float(precision_score(y_test, y_pred) * 100)
    rec = float(recall_score(y_test, y_pred) * 100)
    f1 = float(f1_score(y_test, y_pred) * 100)
    auc = float(roc_auc_score(y_test, y_prob) * 100)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- RANDOM FOREST BENCHMARK SCORES ---")
    print(f"Accuracy  : {acc:.2f}%")
    print(f"Precision : {prec:.2f}%")
    print(f"Recall    : {rec:.2f}%")
    print(f"F1-Score  : {f1:.2f}%")
    print(f"ROC-AUC   : {auc:.2f}%")

    # 1. Metrics & Parameters
    metrics_summary = {
        'model_name': 'Random Forest Classifier',
        'dataset_size': 50000,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(rec, 2),
        'f1_score': round(f1, 2),
        'roc_auc': round(auc, 2),
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(METRICS_DIR, 'random_forest_metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    pd.DataFrame([metrics_summary]).to_csv(os.path.join(METRICS_DIR, 'metrics.csv'), index=False)

    params_df = pd.DataFrame([{
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'criterion': 'gini'
    }])
    params_df.to_csv(os.path.join(METRICS_DIR, 'random_forest_parameters.csv'), index=False)

    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Stable (0)', 'Readmit (1)'],
                yticklabels=['Stable (0)', 'Readmit (1)'])
    plt.title("Random Forest - Validation Confusion Matrix", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Outcome")
    plt.ylabel("Actual Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # 3. Feature Importance
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fi_df.to_csv(os.path.join(FI_DIR, 'feature_importance.csv'), index=False)

    plt.figure(figsize=(9, 6))
    top10_fi = fi_df.head(10)
    sns.barplot(x=top10_fi['Importance'], y=top10_fi['Feature'], palette='crest')
    plt.title("Top 10 Risk Drivers (Random Forest Feature Importance)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FI_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # 4. Charts: Performance Graphs & ROC
    plt.figure(figsize=(7, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='#2f7855', lw=2, label=f'Random Forest (AUC = {auc:.1f}%)')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.title("ROC Characteristic Curve", fontsize=11, fontweight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'performance_graphs.png'), dpi=150)
    plt.close()

    # 5. Persist Model
    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
    # Also update models/best_model.pkl for real-time web inference
    joblib.dump(rf, os.path.join(BASE_DIR, 'models', 'best_model.pkl'))

    # 6. Save Predictions
    pred_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Risk_Probability': np.round(y_prob, 4)
    })
    pred_df.head(500).to_csv(os.path.join(PRED_DIR, 'test_predictions.csv'), index=False)

    print(f"[[OK]] Random Forest fully exported with all subdirectories to: {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_summary

if __name__ == '__main__':
    run_random_forest()

