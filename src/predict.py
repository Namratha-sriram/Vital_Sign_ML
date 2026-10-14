"""Inference helpers for single-patient and uploaded-dataset predictions."""

import os
from functools import lru_cache
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@lru_cache(maxsize=1)
def get_model_and_artifacts():
    """Load the trained estimator and preprocessing artifacts once per process."""
    try:
        import joblib
        paths = [os.path.join(MODELS_DIR, name) for name in ('best_model.pkl', 'scaler.pkl', 'scaled_cols.pkl', 'feature_cols.pkl')]
        if not all(os.path.isfile(path) for path in paths):
            return None, None, None, None
        return tuple(joblib.load(path) for path in paths)
    except Exception:
        return None, None, None, None


def _format_risk_result(prediction, probability):
    prediction = int(prediction)
    probability = float(probability)
    if probability >= 70:
        risk_level, badge_class = 'High Risk / Critical', 'danger'
        recommendations = [
            'Immediate continuous vitals monitoring recommended (SpO2 & ECG).',
            'Schedule specialist consultation and HbA1c / metabolic panel recheck.',
            'Formulate 30-day post-discharge transitional care and medication reconciliation plan.'
        ]
    elif probability >= 40:
        risk_level, badge_class = 'Moderate Risk', 'warning'
        recommendations = [
            'Frequent outpatient follow-up within 7-14 days.',
            'Reinforce glycemic and blood pressure medication adherence.',
            'Educate patient and caregiver on red-flag symptoms.'
        ]
    else:
        risk_level, badge_class = 'Low Risk / Stable', 'success'
        recommendations = [
            'Patient vital signs and clinical biomarkers are within normal recovery thresholds.',
            'Routine primary care follow-up in 30 days.',
            'Continue standard lifestyle and dietary maintenance guidelines.'
        ]
    return {
        'prediction': prediction,
        'probability': round(probability, 1),
        'risk_level': risk_level,
        'badge_class': badge_class,
        'recommendations': recommendations,
    }


def predict_patients_risk(patient_dicts):
    """Predict a sequence of patient records using one vectorized model call."""
    if not patient_dicts:
        return []
    model, scaler, scaled_cols, feature_cols = get_model_and_artifacts()
    if model is None:
        raise RuntimeError(
            'Trained prediction model files are missing or unreadable. '
            'Run src/train_models.py and try again.'
        )
    else:
        frame = pd.DataFrame(patient_dicts)
        categorical_defaults = {
            'gender': 'Female', 'admission_type_id': 1,
            'discharge_disposition_id': 1, 'admission_source_id': 1,
            'max_glu_serum': 'None', 'A1Cresult': 'None', 'insulin': 'No',
        }
        for column, default in categorical_defaults.items():
            if column not in frame:
                frame[column] = default
            else:
                frame[column] = frame[column].fillna(default)
        categorical = list(categorical_defaults)
        frame = pd.get_dummies(
            frame, columns=[column for column in categorical if column in frame.columns],
            drop_first=True
        )
        for column in feature_cols:
            if column not in frame.columns:
                frame[column] = 0
        frame = frame[feature_cols]
        if scaled_cols:
            # Promote integer CSV columns before assigning fractional z-scores.
            numeric_frame = frame[scaled_cols].astype('float64')
            frame[scaled_cols] = scaler.transform(numeric_frame).astype('float64')
        predictions = model.predict(frame)
        if hasattr(model, 'predict_proba'):
            class_probabilities = model.predict_proba(frame)
            classes = list(getattr(model, 'classes_', [0, 1]))
            positive_index = classes.index(1) if 1 in classes else len(classes) - 1
            output = [
                (int(prediction), float(probabilities[positive_index]) * 100)
                for prediction, probabilities in zip(predictions, class_probabilities)
            ]
        elif hasattr(model, 'decision_function'):
            import numpy as np
            decision_scores = np.asarray(model.decision_function(frame), dtype=float)
            probabilities = 100.0 / (1.0 + np.exp(-np.clip(decision_scores, -30, 30)))
            output = [(int(prediction), float(probability))
                      for prediction, probability in zip(predictions, probabilities)]
        else:
            raise RuntimeError('The trained model does not provide usable confidence scores.')
    return [_format_risk_result(prediction, probability)
            for prediction, probability in output]


def predict_patient_risk(patient_dict):
    """Return the standard risk assessment payload for a single patient."""
    return predict_patients_risk([patient_dict])[0]
