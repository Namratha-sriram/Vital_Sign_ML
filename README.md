# VitalSign: Machine Learning-Based Clinical Risk & Readmission Prediction

A complete, end-to-end Machine Learning healthcare decision support system built for B.Tech Computer Science & Engineering.

---

## 1. Project Directory Structure

```text
vital_sign/
│
├── app.py                                  # Flask Web Application with 11 routes (including Before vs After Comparison)
├── main.py                                 # PyCharm entry-point runner
├── requirements.txt                        # Python dependencies
├── README.md                               # Comprehensive documentation & faculty viva preparation
├── run_app.bat                             # One-click Windows launcher for web interface
├── run_training.bat                        # One-click Windows launcher for model training & comparison
│
├── data/
│   ├── IDs_mapping.csv                     # Clinical admission, discharge, and source mapping codes
│   └── vitalsign_data.csv                  # Physiological vitals & patient readmission dataset (50,000 Records)
│
├── models/
│   ├── best_model.pkl                      # Serialized production model (Random Forest)
│   ├── scaler.pkl                          # Fitted StandardScaler for numerical vitals
│   ├── model_metrics.json                  # Stored performance scores, confusion matrix & counts
│   ├── before_after_metrics.json           # Faculty comparative benchmark results (Baseline vs Final)
│   └── feature_cols.pkl                    # List of dummy-encoded feature column names
│
├── output/                                 # Milestone artifacts matching SkillMatch_ML architecture
│   ├── Comparison_Before_After/            # ★ Faculty Deliverable: Before vs After Comparisons
│   │   ├── before_after_comparison.csv     # Tabular metric comparison table
│   │   ├── before_after_metrics.json       # JSON evaluation metrics
│   │   ├── faculty_comparison_report.txt   # Official faculty academic report & viva answers
│   │   ├── before_after_metrics_barchart.png   # Comparative bar chart
│   │   ├── before_after_confusion_matrix.png   # Side-by-side confusion matrix
│   │   ├── before_after_roc_curve.png          # Overlaid ROC curve comparison
│   │   └── before_after_preprocessing_dist.png # Feature distribution before vs after Z-score scaling
│   ├── EDA_Analysis_outputs/               # Milestone 1: Missing heatmaps, boxplots, stats
│   ├── Linear_Regression_with_Metrics_M2/  # Milestone 2: Length of stay regression & residual plots
│   ├── Random_Forest_Tree_M3_Outputs/      # Milestone 3: Random Forest models, charts, feature importance
│   ├── Decision_Tree_Outputs/              # Milestone 3: Decision tree structure & metrics
│   ├── Logistic_Regression_Outputs/        # Milestone 3: Baseline classification outputs
│   └── XGBoost_Outputs/                    # Milestone 3: Gradient boosting metrics
│
├── src/
│   ├── Dataset_Load_identify_missing_values_M1.py   # M1: Data ingestion & null heatmap
│   ├── Missing_Values_Duplicates_M1.py              # M1: Missing values & duplicate audit
│   ├── Correlation_Matrix_heatmap_boxplots_M1.py    # M1: Correlation heatmap & vitals boxplots
│   ├── data_understand.py                           # M1: Demographic & vitals statistical summary
│   ├── EDA Analysis.py                              # M1: Univariate distribution histograms
│   ├── 01_label_encoding.py                         # M2: Label encoding for categorical variables
│   ├── 02_ordinal_encoding.py                       # M2: Ordinal clinical severity encoding
│   ├── 03_embedding_encoding.py                     # M2: Frequency & entity embeddings
│   ├── 04_one_hot_encoding.py                       # M2: Dummy one-hot encoding
│   ├── 05_target_encoding.py                        # M2: Supervised Bayesian target encoding
│   ├── clean_del_mean_model_missing_imputer_M2.py   # M2: Imputation comparison (mean/median/drop)
│   ├── clean_minmax_stand_norma_M2.py               # M2: Feature scaling comparison
│   ├── Linear_Regression_with_Metrics_M2.py         # M2: Hospital stay regression & residual plots
│   ├── final_preprocessor.py                        # M2: Unified ColumnTransformer pipeline
│   ├── logistic_regression_binary_classify.py       # M3: Baseline logistic classifier
│   ├── multinomial_logistic_regre_for_multiclass.py # M3: 3-tier clinical triage classification
│   ├── Decision_Tree_Classsifier_M3.py              # M3: Decision Tree with depth pruning
│   ├── Random_Forest_Tree_M3.py                     # M3: Production Random Forest classifier
│   ├── Adaboost_classifier_M3.py                    # M3: AdaBoost ensemble boosting
│   ├── XGBoost.py / LightGBM.py                     # M3: Gradient boosted decision trees
│   ├── compare_before_after_M3.py                   # ★ Faculty Comparison: Baseline vs Final Model
│   ├── train_models.py                              # Master script training all 5 classifiers
│   └── predict.py                                   # Real-time inference function for patient risk assessment
│
├── static/
│   ├── style.css                           # Clean, editorial ledger UI styling (SkillMatch_ML inspired)
│   └── images/                             # High-resolution generated charts
│       ├── before_after_metrics_barchart.png
│       ├── before_after_confusion_matrix.png
│       ├── before_after_roc_curve.png
│       ├── before_after_preprocessing_dist.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── feature_importance.png
│
└── templates/
    ├── base.html                           # Left rail navigation (01 Home to 11 Contact)
    ├── home.html                           # Landing page with KPIs & comparison highlight
    ├── about.html                          # Problem statement & B.Tech syllabus mapping
    ├── dataset.html                        # 50,000 records preview & data dictionary
    ├── preprocessing.html                  # Preprocessing steps & scaling comparison
    ├── visualization.html                  # EDA and exploratory charts gallery
    ├── models.html                         # Algorithm benchmarks & Before vs After banner
    ├── comparison.html                     # ★ Faculty Showcase: Before vs After Evaluation
    ├── prediction.html                     # Interactive patient risk assessment form
    ├── dashboard.html                      # Hospital surveillance & telemetry tracking
    ├── reports.html                        # Clinical export cards, matrix breakdown & audit
    └── contact.html                        # Academic credentials & project profile
```

---

## 2. Faculty Requirement: Before vs After Comparison

Per faculty instructions, a direct quantitative comparison between **Baseline Results (Before)** and **Optimized Results (After)** was conducted on the full **50,000 patient dataset** (evaluated on 10,000 unseen validation records):

| Evaluation Metric | Before (Baseline Logistic Reg.) | After (Optimized Random Forest) | Improvement (Delta) | Clinical Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 74.04% | **77.72%** | **+3.68%** | Higher overall diagnostic precision across cohort |
| **Precision** | 75.54% | **77.89%** | **+2.36%** | Fewer false alarms for hospital clinical teams |
| **Recall (Sensitivity)** | 77.42% | **82.53%** | **+5.10%** | Detects deteriorating patients before discharge |
| **F1-Score** | 76.47% | **80.14%** | **+3.67%** | Optimal harmonic balance for clinical cohorts |
| **ROC-AUC** | 82.01% | **85.86%** | **+3.85%** | Greater discriminability across decision thresholds |
| **Log Loss** | 0.5157 | **0.4970** | **-0.0187** | Calibrated, higher-confidence probability output |
| **False Negatives (Missed)**| 1,230 | **952** | **-278 (-22.6%)** | **278 fewer missed high-risk patients** (Life-saving) |

### Key Diagnostic Visualizations:
1. `before_after_metrics_barchart.png`: Grouped comparative bar chart of all metrics.
2. `before_after_confusion_matrix.png`: Shows drop in missed critical patients from 1,230 to 952.
3. `before_after_roc_curve.png`: Overlaid ROC curves expanding from 82.0% to 85.9%.
4. `before_after_preprocessing_dist.png`: Glucose standardized from raw [75, 290] mg/dL to Z-score.

---

## 3. Quickstart Guide in PyCharm (Windows)

### Step 1: Open Project in PyCharm
1. Open **PyCharm**.
2. Click **File** &rarr; **Open...** &rarr; Navigate to `C:\Users\namra\PycharmProjects\vital_sign` &rarr; Click **Trust Project**.

### Step 2: Install Python Dependencies
Use Python 3.11 or newer and install the packages from the project root:
```powershell
py -3 -m pip install -r requirements.txt
```

### Step 3: Run Training & Comparison (Generates All Models & Plots)
Run in terminal or double-click `run_training.bat`:
```powershell
python src/train_models.py
python src/compare_before_after_M3.py
```

### Step 4: Launch Web Application
Run in terminal or double-click `run_app.bat`:
```powershell
python app.py
```
Open browser at: `http://127.0.0.1:5000`

---

## 4. Academic Viva Defense Guide

1. **Why did the model improve after optimization?**
   - Baseline Logistic Regression assumes linear boundaries in log-odds space, failing to capture complex non-linear vitals interactions.
   - The Optimized Random Forest aggregates 100 decorrelated trees via bagging, reducing variance and capturing non-linear interactions (e.g., moderate glucose elevation combined with acute hypoxia).
2. **Why is False Negative reduction critical in clinical systems?**
   - False Negatives mean deteriorating patients are mistakenly sent home, risking severe complications or death. The optimized model reduced missed cases by 22.6% (278 patients).
3. **Why normalize continuous vitals?**
   - Raw features have widely varying ranges (Blood Glucose up to 290 mg/dL vs SpO2 89–100%). Z-score scaling (StandardScaler) eliminates scale bias and stabilizes convergence.
