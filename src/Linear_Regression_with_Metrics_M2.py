"""
Linear_Regression_with_Metrics_M2.py
Milestone 2: Linear Regression Model with Diagnostic Metrics
Predicts continuous hospital length of stay (time_in_hospital) from clinical vitals and computes MAE, MSE, RMSE, R2, and residual plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'vitalsign_data.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'Linear_Regression_with_Metrics_M2')
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

def run_linear_regression():
    print("=" * 70)
    print("  MILESTONE 2: LINEAR REGRESSION WITH CLINICAL METRICS (50,000 RECORDS)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, nrows=50000)

    feature_cols = [
        'age', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'blood_glucose', 'respiratory_rate', 'body_temperature',
        'oxygen_saturation', 'num_lab_procedures', 'num_medications'
    ]
    target_col = 'time_in_hospital'

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Compute Regression Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- LINEAR REGRESSION PERFORMANCE METRICS ---")
    print(f"Mean Absolute Error (MAE)  : {mae:.4f} days")
    print(f"Mean Squared Error (MSE)   : {mse:.4f}")
    print(f"Root Mean Squared Error    : {rmse:.4f} days")
    print(f"R-squared Score (R2)       : {r2:.4f}")

    # Feature Coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print("\n--- FEATURE COEFFICIENTS ---")
    print(coef_df.to_string(index=False))

    # Export metrics CSV
    metrics_df = pd.DataFrame([{
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R2_Score': round(r2, 4)
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'linear_regression_metrics.csv'), index=False)
    coef_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_coefficients.csv'), index=False)

    # Generate Residual Plot
    plt.figure(figsize=(8, 5))
    residuals = y_test - y_pred
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, color='#35524c')
    plt.axhline(0, color='#b23a2e', linestyle='--')
    plt.title("Linear Regression - Residuals vs Predicted Length of Stay", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Days in Hospital")
    plt.ylabel("Residuals")
    plt.tight_layout()
    res_img = os.path.join(IMG_DIR, 'residual_plot.png')
    plt.savefig(res_img, dpi=150)
    plt.close()
    print(f"[[OK]] Saved Residual Plot to: {res_img}")
    print("=" * 70)

if __name__ == '__main__':
    run_linear_regression()

