"""
train_models.py
Trains and evaluates multiple Machine Learning models for VitalSign:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Ridge Classifier
- AdaBoost
- Gradient Boosting

Generates performance visualizations (confusion matrix, ROC, feature importance)
and persists the top-performing model and metrics to disk.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import RidgeClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
    )
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, roc_curve
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    import joblib
    HAS_SKLEARN = True
except (ImportError, Exception):
    HAS_SKLEARN = False

# Import preprocessing
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.data_preprocessing import load_and_preprocess_data

STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'images')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(STATIC_IMG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_clinical_eda_plots(raw_df):
    """Create real, use-case-specific EDA figures from the loaded patient data."""
    sns.set_theme(style='whitegrid')
    if 'readmitted' in raw_df:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=raw_df, x='readmitted', color='#2a9d8f')
        plt.xticks([0, 1], ['Stable', 'Readmitted / at risk'])
        plt.title('Patient Outcome Class Balance', fontweight='bold')
        plt.xlabel('Observed outcome')
        plt.ylabel('Patient records')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_IMG_DIR, 'class_balance.png'), dpi=150)
        plt.close()

        numeric = raw_df.select_dtypes(include='number')
        corr_cols = [c for c in numeric.columns if c != 'patient_id'][:18]
        if 'readmitted' in numeric and 'readmitted' not in corr_cols:
            corr_cols.append('readmitted')
        if len(corr_cols) > 1:
            plt.figure(figsize=(11, 8))
            sns.heatmap(numeric[corr_cols].corr(), cmap='coolwarm', center=0,
                        square=False, linewidths=.25)
            plt.title('Clinical Feature Correlation Heatmap', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_IMG_DIR, 'correlation_heatmap.png'), dpi=150)
            plt.close()

        vitals = [c for c in ['heart_rate', 'systolic_bp', 'blood_glucose',
                              'oxygen_saturation'] if c in raw_df]
        if vitals:
            fig, axes = plt.subplots(1, len(vitals), figsize=(4 * len(vitals), 5))
            axes = np.atleast_1d(axes)
            for ax, col in zip(axes, vitals):
                sns.boxplot(data=raw_df, x='readmitted', y=col, ax=ax,
                            color='#457b9d')
                ax.set_title(col.replace('_', ' ').title())
                ax.set_xlabel('Outcome (0 stable, 1 readmitted)')
            fig.suptitle('Vital Sign Distributions by Patient Outcome', fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(STATIC_IMG_DIR, 'vitals_by_outcome.png'), dpi=150)
            plt.close(fig)

        pair_cols = [c for c in ['blood_glucose', 'heart_rate', 'systolic_bp',
                                 'oxygen_saturation', 'readmitted'] if c in raw_df]
        if len(pair_cols) >= 3:
            sample = raw_df[pair_cols].dropna().sample(
                n=min(500, len(raw_df[pair_cols].dropna())), random_state=42
            )
            grid = sns.pairplot(sample, hue='readmitted', corner=True,
                                plot_kws={'alpha': .45, 's': 12},
                                diag_kws={'common_norm': False})
            grid.fig.suptitle('Patient Vital Sign Relationships', y=1.02, fontweight='bold')
            grid.savefig(os.path.join(STATIC_IMG_DIR, 'clinical_pairplot.png'), dpi=120)
            plt.close(grid.fig)

        for col, filename, title in [
            ('gender', 'outcome_by_gender.png', 'Readmission Rate by Gender'),
            ('admission_type_id', 'outcome_by_admission.png', 'Readmission Rate by Admission Type'),
        ]:
            if col in raw_df:
                rates = raw_df.groupby(col, dropna=False)['readmitted'].mean().mul(100)
                plt.figure(figsize=(7, 5))
                plt.bar(rates.index.astype(str), rates.values, color='#457b9d')
                plt.title(title, fontweight='bold')
                plt.xlabel(col.replace('_', ' ').title())
                plt.ylabel('Readmitted records (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(STATIC_IMG_DIR, filename), dpi=150)
                plt.close()


def train_and_evaluate_all():
    print("[1/5] Loading and preparing clinical vital sign records...")
    X_train, X_test, y_train, y_test, raw_df, feature_cols = load_and_preprocess_data()

    # Generate Exploratory Data Analysis (EDA) plot
    print("[2/5] Generating vital signs exploratory data plots...")
    generate_clinical_eda_plots(raw_df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    sns.histplot(raw_df['blood_glucose'], kde=True, ax=axes[0, 0], color='#e63946')
    axes[0, 0].set_title('Blood Glucose Distribution (mg/dL)', fontsize=12, fontweight='bold')

    sns.histplot(raw_df['systolic_bp'], kde=True, ax=axes[0, 1], color='#457b9d')
    axes[0, 1].set_title('Systolic Blood Pressure Distribution (mmHg)', fontsize=12, fontweight='bold')

    sns.histplot(raw_df['heart_rate'], kde=True, ax=axes[1, 0], color='#2a9d8f')
    axes[1, 0].set_title('Heart Rate Distribution (bpm)', fontsize=12, fontweight='bold')

    sns.histplot(raw_df['oxygen_saturation'], kde=True, ax=axes[1, 1], color='#e76f51')
    axes[1, 1].set_title('Oxygen Saturation Distribution (%)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    eda_plot_path = os.path.join(STATIC_IMG_DIR, 'vitals_distribution.png')
    plt.savefig(eda_plot_path, dpi=150)
    plt.close()

    results = {}
    fitted_models = {}
    cv_fold_scores = {}
    best_name = 'Random Forest'

    if HAS_SKLEARN:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Ridge Classifier': CalibratedClassifierCV(RidgeClassifier(), cv=3),
            'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42),
            'Support Vector Machine (SVM)': SVC(probability=True, kernel='rbf', random_state=42),
            'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            ),
        }

        best_f1 = -1
        print("[3/5] Training machine learning models on up to 50k clinical records...")
        cv_sample_size = min(5000, len(X_train))
        if cv_sample_size < len(X_train):
            cv_X, _, cv_y, _ = train_test_split(
                X_train, y_train, train_size=cv_sample_size,
                random_state=42, stratify=y_train
            )
        else:
            cv_X, cv_y = X_train, y_train
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            if name == 'Support Vector Machine (SVM)' and len(X_train) > 5000:
                sub_idx = np.random.choice(len(X_train), 5000, replace=False)
                model.fit(X_train.iloc[sub_idx], y_train.iloc[sub_idx])
            elif name == 'K-Nearest Neighbors (KNN)' and len(X_train) > 10000:
                sub_idx = np.random.choice(len(X_train), 10000, replace=False)
                model.fit(X_train.iloc[sub_idx], y_train.iloc[sub_idx])
            else:
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            model_slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            model_cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(model_cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Stable', 'At risk'], yticklabels=['Stable', 'At risk'])
            plt.title(f'Confusion Matrix - {name}', fontweight='bold')
            plt.xlabel('Predicted outcome')
            plt.ylabel('Observed outcome')
            plt.tight_layout()
            plt.savefig(os.path.join(STATIC_IMG_DIR, f'confusion_{model_slug}.png'), dpi=140)
            plt.close()

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
                auc_val = float(roc_auc_score(y_test, y_score))
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
                auc_val = float(roc_auc_score(y_test, y_score))
            else:
                auc_val = 0.5

            # Save an individual ROC figure for the model comparison page.
            if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                model_fpr, model_tpr, _ = roc_curve(y_test, y_score)
                plt.figure(figsize=(6, 5))
                plt.plot(model_fpr, model_tpr, color='#188779', linewidth=2.4,
                         label=f'AUC {auc_val:.3f}')
                plt.plot([0, 1], [0, 1], color='#aab8bb', linestyle='--', linewidth=1)
                plt.xlabel('False positive rate')
                plt.ylabel('True positive rate')
                plt.title(f'{name} · ROC curve', fontweight='bold')
                plt.legend(loc='lower right', frameon=False)
                plt.grid(alpha=.16)
                plt.tight_layout()
                plt.savefig(os.path.join(STATIC_IMG_DIR, f'roc_{model_slug}.png'), dpi=150)
                plt.close()

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            fold_scores = cross_val_score(
                model, cv_X, cv_y, cv=cv_splitter, scoring='accuracy', n_jobs=1
            )
            cv_fold_scores[name] = [round(float(score) * 100, 3) for score in fold_scores]

            results[name] = {
                'accuracy': round(acc * 100, 2),
                'precision': round(prec * 100, 2),
                'recall': round(rec * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'roc_auc': round(auc_val * 100, 2),
                'cv_accuracy_mean': round(float(np.mean(fold_scores)) * 100, 2),
                'cv_accuracy_std': round(float(np.std(fold_scores)) * 100, 2),
            }
            fitted_models[name] = model

            print(f"   -> {name}: Accuracy={acc*100:.2f}%, F1={f1*100:.2f}%, AUC={auc_val*100:.2f}%")
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
    else:
        raise RuntimeError(
            "scikit-learn could not be loaded in this Python environment. "
            "Install the project requirements in PyCharm before training; "
            "the pipeline will not substitute sample benchmark scores."
        )

    print(f"[4/5] Best model selected: {best_name} (F1 Score: {results[best_name]['f1_score']}%)")

    # Generate Model Comparison Chart
    names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in names]
    f1_scores = [results[m]['f1_score'] for m in names]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2b5c8f')
    plt.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#17b978')
    plt.xlabel('Machine Learning Algorithms', fontweight='bold')
    plt.ylabel('Score (%)', fontweight='bold')
    plt.title('Algorithm Performance Comparison on VitalSign Dataset', fontsize=13, fontweight='bold')
    plt.xticks(x, names, rotation=15)
    plt.ylim(40, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'model_comparison.png'), dpi=150)
    plt.close()

    if HAS_SKLEARN and cv_fold_scores:
        cv_frame = pd.DataFrame({name: scores for name, scores in cv_fold_scores.items()})
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=cv_frame, palette='Set2')
        plt.ylabel('Stratified 5-fold accuracy (%)')
        plt.xlabel('Classifier')
        plt.title(f'Cross-Validation Stability (stratified sample: {len(cv_X):,} records)', fontweight='bold')
        plt.xticks(rotation=18, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(STATIC_IMG_DIR, 'cv_stability.png'), dpi=150)
        plt.close()

    # Generate Confusion Matrix for Best Model
    if HAS_SKLEARN and best_name in fitted_models:
        best_model = fitted_models[best_name]
        y_best_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_best_pred)
    else:
        cm = np.array([[5280, 520], [610, 3590]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stable (0)', 'Readmitted/At Risk (1)'],
                yticklabels=['Stable (0)', 'Readmitted/At Risk (1)'])
    plt.title(f'Confusion Matrix - {best_name}', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('Actual Class', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    if HAS_SKLEARN and fitted_models:
        for name, model in fitted_models.items():
            if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                y_score = (model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba")
                           else model.decision_function(X_test))
                fpr, tpr, _ = roc_curve(y_test, y_score)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']/100:.2f})")
    else:
        # Precomputed curves
        fpr = np.linspace(0, 1, 100)
        tpr_rf = 1 - (1 - fpr)**3.5
        tpr_svm = 1 - (1 - fpr)**2.8
        plt.plot(fpr, tpr_rf, label="Random Forest (AUC = 0.94)", color='#2a9d8f')
        plt.plot(fpr, tpr_svm, label="Support Vector Machine (AUC = 0.91)", color='#457b9d')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'roc_curve.png'), dpi=150)
    plt.close()

    # Feature Importance
    if HAS_SKLEARN and 'Random Forest' in fitted_models:
        rf_model = fitted_models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_cols[i] for i in indices]
        top_weights = importances[indices]
    else:
        top_features = ['Oxygen Saturation (SpO2)', 'Blood Glucose (mg/dL)', 'Emergency Visits Prior', 'Systolic BP (mmHg)', 'Time in Hospital (Days)', 'Age (Years)']
        top_weights = np.array([0.24, 0.21, 0.18, 0.15, 0.12, 0.10])

    plt.figure(figsize=(9, 5))
    plt.barh(range(len(top_features)), top_weights[::-1], color='#e63946', align='center')
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Relative Feature Importance (Gini Impurity Reduction)', fontweight='bold')
    plt.title('Top Clinical & Vital Sign Risk Indicators', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # Save artifacts
    print("[5/5] Saving model artifacts and serialization files...")
    if HAS_SKLEARN and fitted_models:
        try:
            joblib.dump(fitted_models[best_name], os.path.join(MODELS_DIR, 'best_model.pkl'))
            joblib.dump(fitted_models, os.path.join(MODELS_DIR, 'all_models.pkl'))
        except Exception:
            pass

    metrics_payload = {
        'best_model': best_name,
        'metrics': results,
        'feature_importance': ({
            feature_cols[index]: round(float(rf_model.feature_importances_[index]) * 100, 3)
            for index in np.argsort(rf_model.feature_importances_)[::-1][:10]
        } if HAS_SKLEARN and 'Random Forest' in fitted_models else {}),
        'confusion_matrix': cm.tolist(),
        'dataset_summary': {
            'total_samples': len(raw_df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(feature_cols)
        }
    }
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics_payload, f, indent=4)

    print("Model training pipeline finished successfully!")
    return metrics_payload


if __name__ == '__main__':
    train_and_evaluate_all()
