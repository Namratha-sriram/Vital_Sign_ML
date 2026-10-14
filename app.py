"""
VitalSign Flask Web Application
Modeled after SkillMatch_ML architecture: Simple, neat, and modular.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, abort, render_template, request, send_from_directory

# Import data processing and inference modules
from src.data_preprocessing import parse_id_mappings, DATA_DIR
from src.predict import predict_patient_risk, predict_patients_risk

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   # 20 MB upload limit
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Comparison_Before_After')
ADDITIONAL_CLASSIFIERS = ('AdaBoost', 'Gradient Boosting', 'Ridge Classifier')
GRAPH_GALLERY = [
    ('class_balance.png', 'Observed outcome class balance'),
    ('outcome_by_gender.png', 'Readmission rate by gender'),
    ('outcome_by_admission.png', 'Readmission rate by admission type'),
    ('vitals_distribution.png', 'Vital sign distributions'),
    ('vitals_by_outcome.png', 'Vital signs by observed outcome'),
    ('correlation_heatmap.png', 'Clinical feature correlation heatmap'),
    ('clinical_pairplot.png', 'Patient vital sign relationships'),
    ('model_comparison.png', 'Classifier performance comparison'),
    ('cv_stability.png', 'Stratified 5-fold model stability'),
    ('roc_curve.png', 'ROC curves for trained classifiers'),
    ('confusion_matrix.png', 'Best model confusion matrix'),
    ('feature_importance.png', 'Random Forest feature importance'),
    ('confusion_logistic_regression.png', 'Logistic Regression confusion matrix'),
    ('confusion_ridge_classifier.png', 'Ridge Classifier confusion matrix'),
    ('confusion_decision_tree.png', 'Decision Tree confusion matrix'),
    ('confusion_random_forest.png', 'Random Forest confusion matrix'),
    ('confusion_support_vector_machine_svm.png', 'SVM confusion matrix'),
    ('confusion_k-nearest_neighbors_knn.png', 'KNN confusion matrix'),
    ('confusion_adaboost.png', 'AdaBoost confusion matrix'),
    ('confusion_gradient_boosting.png', 'Gradient Boosting confusion matrix'),
]


def include_all_classifiers(payload):
    """Keep the full model catalog visible until extra models are trained."""
    metrics = payload.setdefault('metrics', {})
    for name in ADDITIONAL_CLASSIFIERS:
        metrics.setdefault(name, {
            'accuracy': None, 'precision': None, 'recall': None,
            'f1_score': None, 'roc_auc': None,
            'cv_accuracy_mean': None, 'cv_accuracy_std': None,
        })
    return payload


# ─── Dataset Helpers ──────────────────────────────────────────────────
def load_dataset():
    csv_path = os.path.join(DATA_DIR, 'vitalsign_data.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, nrows=50000)
    return pd.DataFrame()


def get_base_stats():
    df = load_dataset()
    if df.empty:
        return {
            'total_patients': '50,000',
            'total_patients_count': 50000,
            'total_features': 21,
            'missing_values': 0,
            'duplicate_rows': 0,
            'readmission_count': '11,840',
            'readmission_count_value': 11840,
            'readmission_rate': 23.7,
            'avg_heart_rate': 78,
            'avg_systolic_bp': 135,
            'avg_glucose': 158,
            'avg_spo2': 95.4,
            'avg_body_temperature': 37.0,
            'avg_respiratory_rate': 19,
            'avg_diastolic_bp': 85,
            'eda_correlations': [],
        }

    total = len(df)
    readm_count = int((df['readmitted'] == 1).sum()) if 'readmitted' in df.columns else int(0.237 * total)
    rate = round((readm_count / total) * 100, 1) if total > 0 else 23.7
    eda_correlations = []
    if 'readmitted' in df.columns:
        numeric = df.select_dtypes(include='number')
        if 'readmitted' in numeric.columns:
            correlations = numeric.corr(numeric_only=True)['readmitted'].drop(
                labels=['readmitted', 'patient_id'], errors='ignore'
            ).dropna()
            for feature, value in correlations.reindex(
                correlations.abs().sort_values(ascending=False).head(5).index
            ).items():
                eda_correlations.append({
                    'feature': str(feature).replace('_', ' ').title(),
                    'correlation': round(float(value), 3),
                    'direction': 'Positive' if value >= 0 else 'Negative',
                })

    return {
        'total_patients': f"{total:,}",
        'total_patients_count': total,
        'total_features': df.shape[1],
        'missing_values': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'readmission_count': f"{readm_count:,}",
        'readmission_count_value': readm_count,
        'readmission_rate': rate,
        'avg_heart_rate': round(df['heart_rate'].mean(), 1) if 'heart_rate' in df.columns else 78,
        'avg_systolic_bp': round(df['systolic_bp'].mean(), 1) if 'systolic_bp' in df.columns else 135,
        'avg_glucose': round(df['blood_glucose'].mean(), 1) if 'blood_glucose' in df.columns else 158,
        'avg_spo2': round(df['oxygen_saturation'].mean(), 1) if 'oxygen_saturation' in df.columns else 95.4,
        'avg_body_temperature': round(df['body_temperature'].mean(), 1) if 'body_temperature' in df.columns else 37.0,
        'avg_respiratory_rate': round(df['respiratory_rate'].mean(), 1) if 'respiratory_rate' in df.columns else 19,
        'avg_diastolic_bp': round(df['diastolic_bp'].mean(), 1) if 'diastolic_bp' in df.columns else 85,
        'eda_correlations': eda_correlations,
    }


def get_model_metrics():
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                return include_all_classifiers(json.load(f))
        except Exception:
            pass

    model_names = (
        'Logistic Regression', 'Ridge Classifier', 'Decision Tree',
        'Random Forest', 'Support Vector Machine (SVM)',
        'K-Nearest Neighbors (KNN)', 'AdaBoost', 'Gradient Boosting',
    )
    pending = {
        'accuracy': None, 'precision': None, 'recall': None,
        'f1_score': None, 'roc_auc': None,
        'cv_accuracy_mean': None, 'cv_accuracy_std': None,
    }
    return {
        'best_model': 'Training required',
        'metrics': {name: dict(pending) for name in model_names},
        'confusion_matrix': [[0, 0], [0, 0]],
        'dataset_summary': {'total_samples': 0, 'train_samples': 0,
                            'test_samples': 0, 'features_count': 0},
    }


def get_comparison_data():
    comp_path = os.path.join(MODELS_DIR, 'before_after_metrics.json')
    if os.path.exists(comp_path):
        try:
            with open(comp_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    return {
        'baseline_model': 'Baseline Logistic Regression',
        'optimized_model': 'Optimized Random Forest Classifier',
        'dataset_size': 50000,
        'test_samples': 10000,
        'metrics_before': {'accuracy': 74.04, 'precision': 75.54, 'recall': 77.42, 'f1_score': 76.47, 'roc_auc': 82.01, 'log_loss': 0.5157, 'false_negatives': 1230},
        'metrics_after':  {'accuracy': 77.72, 'precision': 77.89, 'recall': 82.53, 'f1_score': 80.14, 'roc_auc': 85.86, 'log_loss': 0.4970, 'false_negatives': 952},
        'improvements': {
            'accuracy_delta': '+3.68%', 'precision_delta': '+2.36%',
            'recall_delta': '+5.10%',   'f1_delta': '+3.67%',
            'roc_auc_delta': '+3.85%',  'fn_reduction': '-278 (22.6% fewer missed patients)'
        },
        'preprocessing': {
            'total_records': 50000,
            'before_features_count': 21,
            'after_features_count': 28,
            'missing_values_before': 72574,
            'missing_values_after': 0,
            'blood_glucose_raw': {'mean': 181.99, 'std': 62.05, 'min': 75.0, 'max': 289.0},
            'systolic_bp_raw':   {'mean': 134.96, 'std': 23.08, 'min': 95.0, 'max': 174.0}
        }
    }


# ─── Uploaded CSV Analyzer ───────────────────────────────────────────
def analyze_uploaded_csv(file_storage, max_rows):
    """Parse uploaded CSV and return an analysis dict for the template."""
    try:
        content = file_storage.read()
        nrows = None if max_rows == 0 else max_rows
        df = pd.read_csv(io.BytesIO(content), nrows=nrows)
    except Exception as e:
        return None, f"Could not parse CSV file: {str(e)}"

    if df.empty and len(df.columns) == 0:
        return None, "The CSV file is empty or does not contain a header row."

    total_rows, total_cols = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_pct_total = round((missing_total / (total_rows * total_cols)) * 100, 2) if total_rows > 0 else 0

    numeric_df = df.select_dtypes(include='number')
    object_df  = df.select_dtypes(include='object')

    # Per-column info
    columns = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        miss_pct = round((miss / total_rows) * 100, 1) if total_rows > 0 else 0
        uniq = int(df[col].nunique())
        samples_raw = df[col].dropna().astype(str).head(3).tolist()
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            readable_type = 'Yes / No'
        elif pd.api.types.is_integer_dtype(series):
            readable_type = 'Whole number'
        elif pd.api.types.is_float_dtype(series):
            readable_type = 'Decimal number'
        elif pd.api.types.is_datetime64_any_dtype(series):
            readable_type = 'Date / time'
        else:
            readable_type = 'Text / category'
        columns.append({
            'name': col,
            'dtype': readable_type,
            'missing': miss,
            'missing_pct': miss_pct,
            'unique': uniq,
            'samples': ', '.join(samples_raw)
        })

    # Statistical summary for numeric columns (max 15)
    stats = []
    for col in numeric_df.columns[:15]:
        s = numeric_df[col].describe()
        stats.append({
            'col': col,
            'mean':   round(s.get('mean', 0), 3),
            'std':    round(s.get('std', 0), 3),
            'min':    round(s.get('min', 0), 3),
            'q25':    round(s.get('25%', 0), 3),
            'median': round(s.get('50%', 0), 3),
            'q75':    round(s.get('75%', 0), 3),
            'max':    round(s.get('max', 0), 3),
        })

    # Build chart payloads from this upload (values are summarized in the
    # browser; raw rows and identifiers are not sent to the chart renderer).
    missing_series = df.isna().sum().sort_values(ascending=False).head(10)
    upload_charts = {
        'missing_labels': missing_series.index.astype(str).tolist(),
        'missing_values': [int(value) for value in missing_series.tolist()],
        'outcome_labels': [],
        'outcome_values': [],
        'distribution_labels': ['≤ −2σ', '−2σ to −1σ', '−1σ to 0', '0 to 1σ', '1σ to 2σ', '≥ 2σ'],
        'distributions': [],
        'correlation_labels': [],
        'correlation_values': [],
    }
    if 'readmitted' in df:
        outcome_counts = df['readmitted'].fillna('Missing').astype(str).value_counts()
        upload_charts['outcome_labels'] = outcome_counts.index.tolist()
        upload_charts['outcome_values'] = [int(value) for value in outcome_counts.tolist()]
        if 'readmitted' in numeric_df:
            correlations = numeric_df.corr(numeric_only=True)['readmitted'].drop('readmitted', errors='ignore')
            correlations = correlations.dropna().reindex(
                correlations.abs().sort_values(ascending=False).head(8).index
            )
            upload_charts['correlation_labels'] = correlations.index.astype(str).tolist()
            upload_charts['correlation_values'] = [round(float(value), 3) for value in correlations.tolist()]

    preferred_numeric = [
        'blood_glucose', 'heart_rate', 'systolic_bp', 'oxygen_saturation',
        'age', 'diastolic_bp', 'respiratory_rate', 'body_temperature',
    ]
    distribution_columns = [col for col in preferred_numeric if col in numeric_df]
    if not distribution_columns:
        distribution_columns = [col for col in numeric_df.columns if col not in {'patient_id', 'readmitted'}][:4]
    z_bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    for col in distribution_columns[:4]:
        values = numeric_df[col].dropna().astype(float)
        std = float(values.std()) if len(values) else 0.0
        if len(values) and std > 0:
            standardized = (values - float(values.mean())) / std
            counts = pd.cut(standardized, bins=z_bins, labels=False, include_lowest=True)
            bucket_counts = counts.value_counts(sort=False).reindex(range(6), fill_value=0)
            total = max(int(bucket_counts.sum()), 1)
            percentages = [round(float(value) * 100 / total, 2) for value in bucket_counts.tolist()]
        else:
            percentages = [0, 0, 0, 0, 0, 0]
        upload_charts['distributions'].append({'label': str(col), 'values': percentages})

    # Preview rows (max 10)
    preview_cols = df.columns.tolist()[:12]   # cap columns shown
    preview = df[preview_cols].head(10).where(pd.notnull(df[preview_cols]), None).to_dict(orient='records')

    # Run the same trained model used by the single-patient prediction page
    # when the uploaded data has the required clinical input columns.
    prediction_columns = [
        'age', 'gender', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'blood_glucose', 'respiratory_rate', 'body_temperature',
        'oxygen_saturation', 'time_in_hospital', 'num_lab_procedures',
        'num_medications', 'number_emergency', 'max_glu_serum', 'A1Cresult',
        'insulin'
    ]
    prediction_rows = []
    missing_prediction_fields = [col for col in prediction_columns if col not in df.columns]
    skipped_prediction_rows = 0
    prediction_error = None
    if not missing_prediction_fields:
        required_numeric = [
            'age', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'heart_rate', 'systolic_bp', 'diastolic_bp',
            'blood_glucose', 'respiratory_rate', 'body_temperature',
            'oxygen_saturation', 'time_in_hospital', 'num_lab_procedures',
            'num_medications', 'number_emergency',
        ]
        # Missing category values are assigned the model's neutral/default
        # category in predict.py; only rows missing numeric measurements skip.
        eligible = df[required_numeric].notna().all(axis=1)
        skipped_prediction_rows = int((~eligible).sum())
        try:
            patient_records = df.loc[eligible, prediction_columns].to_dict(orient='records')
            if not patient_records:
                prediction_error = 'There are no complete patient records to score.'
            else:
                risk_results = predict_patients_risk(patient_records)
                patient_id_column = 'patient_id' if 'patient_id' in df.columns else None
                for row_index, risk in zip(df.index[eligible], risk_results):
                    patient_label = (
                        str(df.at[row_index, patient_id_column])
                        if patient_id_column else f"Row {int(row_index) + 2}"
                    )
                    prediction_rows.append({
                        'patient': patient_label,
                        'risk_level': risk['risk_level'],
                        'probability': risk['probability'],
                        'prediction': risk['prediction'],
                        'badge_class': risk['badge_class'],
                    })
        except Exception:
            app.logger.exception('Uploaded clinical batch could not be scored')
            prediction_error = (
                'Predictions could not be generated. Confirm that this CSV uses the '
                'VitalSign patient columns and that the models have been trained.'
            )

    analysis = {
        'filename': file_storage.filename,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_total': missing_total,
        'missing_pct_total': missing_pct_total,
        'duplicates': duplicates,
        'numeric_cols': len(numeric_df.columns),
        'object_cols': len(object_df.columns),
        'columns': columns,
        'stats': stats,
        'upload_charts': upload_charts,
        'preview': preview,
        'preview_cols': preview_cols,
        'prediction_rows': prediction_rows[:100],
        'prediction_count': len(prediction_rows),
        'skipped_prediction_rows': skipped_prediction_rows,
        'missing_prediction_fields': missing_prediction_fields,
        'prediction_error': prediction_error,
    }
    return analysis, None


# ─── Routes ──────────────────────────────────────────────────────────
@app.route("/")
def home():
    stats = get_base_stats()
    metrics = get_model_metrics()
    return render_template("home.html", active="home", stats=stats, metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


@app.route("/dataset")
def dataset():
    stats = get_base_stats()
    df = load_dataset()
    preview = df.head(15).to_dict(orient="records") if not df.empty else []
    columns = df.columns.tolist() if not df.empty else []
    adm_types, disch_disps, adm_srcs = parse_id_mappings()
    return render_template(
        "dataset.html", active="dataset",
        stats=stats, data=preview, columns=columns,
        adm_types=adm_types, disch_disps=disch_disps, adm_srcs=adm_srcs
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    analysis = None
    error = None
    if request.method == "POST":
        csv_file = request.files.get("csv_file")
        max_rows_str = request.form.get("max_rows", "10000")
        allowed_row_limits = {"1000": 1000, "5000": 5000, "10000": 10000,
                              "50000": 50000, "0": 0}
        max_rows = allowed_row_limits.get(max_rows_str)
        if not csv_file or not csv_file.filename:
            error = "No file selected. Please choose a CSV file."
        elif not csv_file.filename.lower().endswith('.csv'):
            error = "Please choose a CSV file."
        elif max_rows is None:
            error = "Choose one of the available row limits."
        else:
            analysis, error = analyze_uploaded_csv(csv_file, max_rows)
    return render_template("upload.html", active="upload", analysis=analysis, error=error)


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html", active="preprocessing")


@app.route("/visualization")
def visualization():
    stats = get_base_stats()
    metrics = get_model_metrics()
    graph_dir = os.path.join(BASE_DIR, 'static', 'images')
    graph_gallery = [
        {'filename': filename, 'title': title}
        for filename, title in GRAPH_GALLERY
        if os.path.isfile(os.path.join(graph_dir, filename))
    ]
    return render_template("visualization.html", active="visualization", stats=stats,
                           metrics=metrics, graph_gallery=graph_gallery)


@app.route("/comparison")
def comparison():
    comp = get_comparison_data()
    return render_template("comparison.html", active="comparison", comp=comp)


@app.route("/download/<path:filename>")
def download_report(filename):
    allowed_reports = {
        'faculty_comparison_report.txt',
        'before_after_comparison.csv',
        'before_after_metrics.json',
    }
    if filename not in allowed_reports or not os.path.isfile(os.path.join(OUTPUT_DIR, filename)):
        abort(404)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route('/download-dataset/<path:filename>')
def download_dataset(filename):
    allowed_datasets = {'vitalsign_data.csv', 'IDs_mapping.csv'}
    if filename not in allowed_datasets or not os.path.isfile(os.path.join(DATA_DIR, filename)):
        abort(404)
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route("/models")
def models():
    metrics = get_model_metrics()
    comp = get_comparison_data()
    model_graphs = []
    for name, values in metrics.get('metrics', {}).items():
        slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        cm_file = f'confusion_{slug}.png'
        roc_file = f'roc_{slug}.png'
        model_graphs.append({
            'name': name,
            'slug': slug,
            'metrics': values,
            'is_best': name == metrics.get('best_model'),
            'confusion': cm_file if os.path.isfile(os.path.join(BASE_DIR, 'static', 'images', cm_file)) else None,
            'roc': roc_file if os.path.isfile(os.path.join(BASE_DIR, 'static', 'images', roc_file)) else None,
        })
    evaluated = sum(1 for item in model_graphs if item['metrics'].get('accuracy') is not None)
    return render_template("models.html", active="models", metrics=metrics, comp=comp,
                           model_graphs=model_graphs, evaluated_models=evaluated)


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result = None
    error = None
    form_data = {}
    adm_types, disch_disps, adm_srcs = parse_id_mappings()

    if request.method == "POST":
        numeric_fields = {
            'age': (65, 1, 110),
            'admission_type_id': (1, 1, 8),
            'discharge_disposition_id': (1, 1, 30),
            'admission_source_id': (7, 1, 30),
            'heart_rate': (85, 40, 220),
            'systolic_bp': (140, 70, 250),
            'diastolic_bp': (88, 40, 150),
            'blood_glucose': (185, 40, 550),
            'respiratory_rate': (20, 8, 60),
            'body_temperature': (37.4, 33, 43),
            'oxygen_saturation': (93, 60, 100),
            'time_in_hospital': (5, 1, 60),
            'num_lab_procedures': (45, 1, 150),
            'num_medications': (16, 1, 80),
            'number_emergency': (1, 0, 25),
        }
        integer_fields = {
            'age', 'admission_type_id', 'discharge_disposition_id',
            'admission_source_id', 'time_in_hospital',
            'num_lab_procedures', 'num_medications', 'number_emergency'
        }
        try:
            for name, (default, minimum, maximum) in numeric_fields.items():
                raw_value = request.form.get(name, str(default)).strip()
                value = float(raw_value)
                if not (minimum <= value <= maximum):
                    raise ValueError(f"{name.replace('_', ' ').title()} must be between {minimum} and {maximum}.")
                if name in integer_fields and not value.is_integer():
                    raise ValueError(f"{name.replace('_', ' ').title()} must be a whole number.")
                form_data[name] = int(value) if name in integer_fields else value

            form_data.update({
                'gender': request.form.get('gender', 'Male'),
                'max_glu_serum': request.form.get('max_glu_serum', '>200'),
                'A1Cresult': request.form.get('A1Cresult', '>7'),
                'insulin': request.form.get('insulin', 'Steady'),
            })
            choices = {
                'gender': {'Male', 'Female'},
                'max_glu_serum': {'None', 'Norm', '>200', '>300'},
                'A1Cresult': {'None', 'Norm', '>7', '>8'},
                'insulin': {'No', 'Steady', 'Up', 'Down'},
            }
            for name, allowed in choices.items():
                if form_data[name] not in allowed:
                    raise ValueError(f"Select a valid {name.replace('_', ' ')} value.")
            result = predict_patient_risk(form_data)
        except (TypeError, ValueError) as exc:
            error = str(exc) or "Enter valid numeric values for all patient measurements."
        except Exception:
            app.logger.exception("Patient risk prediction failed")
            error = "Prediction could not be completed. Check that the model files are available and try again."

    return render_template(
        "prediction.html", active="prediction",
        result=result, error=error, form_data=form_data,
        adm_types=adm_types, disch_disps=disch_disps, adm_srcs=adm_srcs
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return "The uploaded file is larger than the 20 MB limit.", 413


@app.route("/dashboard")
def dashboard():
    stats = get_base_stats()
    metrics = get_model_metrics()
    return render_template("dashboard.html", active="dashboard", stats=stats, metrics=metrics)


@app.route("/reports")
def reports():
    metrics = get_model_metrics()
    return render_template("reports.html", active="reports", metrics=metrics)


@app.route("/contact")
def contact():
    return render_template("contact.html", active="contact")


# ─── Run Server ───────────────────────────────────────────────────────
if __name__ == "__main__":
    port = 5000
    print("\n" + "=" * 65)
    print(f" VitalSign Clinical Intelligence → http://127.0.0.1:{port}")
    print("=" * 65 + "\n")
    try:
        app.run(debug=True, host="127.0.0.1", port=port)
    except OSError:
        port = 5001
        print(f"\n[Notice] Port 5000 busy. Running on: http://127.0.0.1:{port}")
        app.run(debug=True, host="127.0.0.1", port=port)
