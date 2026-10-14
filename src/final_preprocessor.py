"""
final_preprocessor.py
Milestone 2: Unified Final Preprocessor Pipeline for VitalSign 50k Dataset
Applies ColumnTransformer combining StandardScaler on numeric vitals and OneHotEncoder on nominal features.
Persists processed splits and transformer artifacts to models/ and output/.
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def run_final_preprocessor():
    print("=" * 70)
    print("  MILESTONE 2: UNIFIED PIPELINE PREPROCESSOR (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)
    print(f"[+] Loaded raw dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    numeric_features = [
        'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'blood_glucose', 'respiratory_rate', 'body_temperature',
        'oxygen_saturation', 'time_in_hospital', 'num_lab_procedures',
        'num_medications', 'number_emergency'
    ]

    categorical_features = [
        'gender', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'max_glu_serum', 'A1Cresult', 'insulin'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    X = df[numeric_features + categorical_features]
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # Extract clean feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)

    print(f"[[OK]] Pipeline transformed: {X_train_trans.shape[0]:,} train samples, {X_test_trans.shape[0]:,} test samples")
    print(f"[[OK]] Total engineered input features: {len(all_feature_names)}")

    # Persist transformer & metadata
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'final_preprocessor_pipeline.pkl'))
    joblib.dump(all_feature_names, os.path.join(MODELS_DIR, 'pipeline_feature_names.pkl'))

    # Save small processed sample for inspection
    sample_df = pd.DataFrame(X_train_trans[:100], columns=all_feature_names)
    sample_df['readmitted'] = y_train.iloc[:100].values
    sample_df.to_csv(os.path.join(OUTPUT_DIR, 'final_preprocessed_sample.csv'), index=False)

    print(f"[[OK]] Exported preprocessor pipeline & sample to {OUTPUT_DIR}")
    print("=" * 70)
    return preprocessor, all_feature_names

if __name__ == '__main__':
    run_final_preprocessor()

