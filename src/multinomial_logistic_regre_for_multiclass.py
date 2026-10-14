"""
multinomial_logistic_regre_for_multiclass.py
Milestone 3: Multinomial Logistic Regression for Multi-Tier Clinical Risk (50,000 Records)
Classifies patients into 3 triage urgency levels:
0: Low Risk (Routine Discharge)
1: Moderate Risk (Observation / Follow-up)
2: High Risk (Immediate ICU / Telemetry Escalation)
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Multinomial_Logistic_Regression_Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_multinomial_logistic_regression():
    print("=" * 70)
    print("  MILESTONE 3: MULTINOMIAL LOGISTIC REGRESSION (3-TIER CLINICAL RISK)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)

    # Derive 3-tier clinical triage category from vitals
    # 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk
    risk_score = (
        (df['blood_glucose'] > 180).astype(int) +
        (df['systolic_bp'] > 140).astype(int) +
        (df['oxygen_saturation'] < 94).astype(int) * 2 +
        (df['heart_rate'] > 100).astype(int)
    )
    triage_class = pd.cut(risk_score, bins=[-1, 0, 2, 6], labels=[0, 1, 2]).astype(int)

    features = ['age', 'heart_rate', 'systolic_bp', 'diastolic_bp', 'blood_glucose', 'oxygen_saturation', 'respiratory_rate']
    X = df[features]
    y = triage_class

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n[+] Multinomial 3-Class Accuracy: {acc:.2f}%")
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'Moderate Risk', 'High Risk'], output_dict=True)

    metrics_data = {
        'model': 'Multinomial Logistic Regression',
        'accuracy': round(acc, 2),
        'confusion_matrix': cm,
        'classification_report': report
    }
    with open(os.path.join(OUTPUT_DIR, 'multinomial_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[[OK]] Multinomial outputs exported to {OUTPUT_DIR}")
    print("=" * 70)
    return metrics_data

if __name__ == '__main__':
    run_multinomial_logistic_regression()

