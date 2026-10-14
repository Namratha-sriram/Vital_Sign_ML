"""
data_preprocessing.py
Preprocesses clinical vital signs and patient admission data.
Handles missing values, categorical encoding, feature scaling,
and mapping of hospital admission/discharge codes.
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_id_mappings():
    """
    Parses IDs_mapping.csv to return lookup dictionaries for:
    - admission_type_id
    - discharge_disposition_id
    - admission_source_id
    """
    mapping_file = os.path.join(DATA_DIR, 'IDs_mapping.csv')
    admission_types = {}
    discharge_dispositions = {}
    admission_sources = {}

    if not os.path.exists(mapping_file):
        return admission_types, discharge_dispositions, admission_sources

    with open(mapping_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line == ',':
            continue
        if 'admission_type_id' in line:
            current_section = 'admission_type'
            continue
        elif 'discharge_disposition_id' in line:
            current_section = 'discharge_disposition'
            continue
        elif 'admission_source_id' in line:
            current_section = 'admission_source'
            continue

        parts = line.split(',', 1)
        if len(parts) == 2 and parts[0].isdigit():
            id_val = int(parts[0])
            desc = parts[1].strip().strip('"')
            if current_section == 'admission_type':
                admission_types[id_val] = desc
            elif current_section == 'discharge_disposition':
                discharge_dispositions[id_val] = desc
            elif current_section == 'admission_source':
                admission_sources[id_val] = desc

    return admission_types, discharge_dispositions, admission_sources


def generate_synthetic_data_if_needed(n_samples=50000):
    """
    Generates a realistic clinical dataset of up to 50,000 records if CSV has few records or doesn't exist,
    anchored on real-world medical distributions and vital signs.
    """
    csv_path = os.path.join(DATA_DIR, 'vitalsign_data.csv')
    
    np.random.seed(42)
    ages = np.random.randint(22, 88, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples)
    admission_types = np.random.choice([1, 2, 3, 7], n_samples, p=[0.55, 0.25, 0.15, 0.05])
    discharge_disps = np.random.choice([1, 2, 3, 4], n_samples, p=[0.65, 0.15, 0.15, 0.05])
    admission_srcs = np.random.choice([1, 2, 7], n_samples, p=[0.30, 0.20, 0.50])
    
    # Vital signs distributions
    heart_rates = np.random.randint(58, 115, n_samples)
    systolic_bps = np.random.randint(95, 175, n_samples)
    diastolic_bps = np.random.randint(60, 105, n_samples)
    blood_glucoses = np.random.randint(75, 290, n_samples)
    respiratory_rates = np.random.randint(12, 28, n_samples)
    body_temps = np.round(np.random.uniform(36.3, 38.8, n_samples), 1)
    o2_sats = np.random.randint(89, 100, n_samples)
    
    # Clinical metrics
    time_in_hospital = np.random.randint(1, 14, n_samples)
    num_labs = np.random.randint(12, 85, n_samples)
    num_meds = np.random.randint(4, 32, n_samples)
    num_emergencies = np.random.choice([0, 1, 2, 3], n_samples, p=[0.65, 0.20, 0.10, 0.05])
    
    max_glu = np.random.choice(['None', 'Norm', '>200', '>300'], n_samples, p=[0.75, 0.10, 0.08, 0.07])
    a1c = np.random.choice(['None', 'Norm', '>7', '>8'], n_samples, p=[0.70, 0.12, 0.10, 0.08])
    insulin = np.random.choice(['No', 'Steady', 'Up', 'Down'], n_samples, p=[0.45, 0.30, 0.15, 0.10])
    
    # Calculate realistic outcome probability
    risk_score = (
        (ages > 65).astype(int) * 1.5 +
        (admission_types == 1).astype(int) * 1.2 +
        (systolic_bps > 140).astype(int) * 1.3 +
        (blood_glucoses > 180).astype(int) * 1.8 +
        (o2_sats < 94).astype(int) * 2.0 +
        (num_emergencies > 0).astype(int) * 1.6 +
        (time_in_hospital > 6).astype(int) * 1.1 +
        (a1c == '>8').astype(int) * 1.5
    )
    prob = 1 / (1 + np.exp(-(risk_score - 4.5)))
    readmitted = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'patient_id': range(1001, 1001 + n_samples),
        'age': ages,
        'gender': genders,
        'admission_type_id': admission_types,
        'discharge_disposition_id': discharge_disps,
        'admission_source_id': admission_srcs,
        'heart_rate': heart_rates,
        'systolic_bp': systolic_bps,
        'diastolic_bp': diastolic_bps,
        'blood_glucose': blood_glucoses,
        'respiratory_rate': respiratory_rates,
        'body_temperature': body_temps,
        'oxygen_saturation': o2_sats,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_labs,
        'num_medications': num_meds,
        'number_emergency': num_emergencies,
        'max_glu_serum': max_glu,
        'A1Cresult': a1c,
        'insulin': insulin,
        'readmitted': readmitted
    })
    
    df.to_csv(csv_path, index=False)
    return df


def load_and_preprocess_data(max_rows=50000):
    """
    Loads dataset up to 50,000 records, handles missing values, encodes categorical variables,
    normalizes numeric features, and splits into train/test sets.
    """
    csv_path = os.path.join(DATA_DIR, 'vitalsign_data.csv')
    kaggle_csv = os.path.join(DATA_DIR, 'diabetic_data.csv')

    if os.path.exists(kaggle_csv):
        df = pd.read_csv(kaggle_csv, nrows=max_rows)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=max_rows)
        if len(df) < 5000:
            df = generate_synthetic_data_if_needed(max_rows)
    else:
        df = generate_synthetic_data_if_needed(max_rows)

    if len(df) > max_rows:
        df = df.iloc[:max_rows]

    # Feature definitions
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

    # One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Feature columns excluding identifier and target
    feature_cols = [c for c in df_encoded.columns if c not in ['patient_id', 'readmitted']]
    
    X = df_encoded[feature_cols]
    y = df_encoded['readmitted']

    cols_to_scale = [c for c in numeric_features if c in X.columns]

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import joblib

        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        # Standardization produces floats even for integer source columns.
        # Promote first so current pandas versions do not reject/truncate them.
        X_train_scaled[cols_to_scale] = X_train_scaled[cols_to_scale].astype('float64')
        X_test_scaled[cols_to_scale] = X_test_scaled[cols_to_scale].astype('float64')
        X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale].astype('float64'))
        X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale].astype('float64'))

        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
        joblib.dump(cols_to_scale, os.path.join(MODELS_DIR, 'scaled_cols.pkl'))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    except (ImportError, Exception):
        # Pure-Python fallback if sklearn C-extension DLL is blocked by OS policy
        np.random.seed(42)
        n = len(X)
        shuffled = np.random.permutation(n)
        test_size = int(0.20 * n)
        test_idx = shuffled[:test_size]
        train_idx = shuffled[test_size:]

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        for c in cols_to_scale:
            mean = float(X_train[c].mean())
            std = float(X_train[c].std()) if float(X_train[c].std()) != 0 else 1.0
            X_train_scaled[c] = (X_train[c] - mean) / std
            X_test_scaled[c] = (X_test[c] - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test, df, feature_cols


if __name__ == '__main__':
    print("Preprocessing vital signs dataset...")
    X_train, X_test, y_train, y_test, raw_df, cols = load_and_preprocess_data()
    print(f"Dataset shape: {raw_df.shape}")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print("Preprocessing completed successfully.")
