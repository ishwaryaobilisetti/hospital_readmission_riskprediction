# 30-Day Hospital Readmission Risk Predictor

An end-to-end, reproducible, and responsible machine learning system to predict 30-day hospital readmissions for diabetic patients. Built on a decade of clinical encounter records from the Diabetes 130-US Hospitals (1999–2008) dataset, the pipeline addresses real-world data challenges (such as data leakage and class imbalance) and implements rigorous model calibration, SHAP-based local/global explainability, and algorithmic fairness auditing.

---

## 🚀 Key Features

1. **Target Binarization & Leakage Prevention**: Translates patient readmission history into a clean binary classification task (readmission < 30 days) and filters out deceased or hospice patients (discharge disposition IDs 11, 13, 14, 19, 20, 21) who have a zero probability of readmission.
2. **Reproducible Pipeline**: Encapsulates all data cleaning, numerical scaling, missing category imputation, and custom feature engineering inside a single, fully serializable Scikit-learn `Pipeline` object (`pipeline.pkl`).
3. **Custom Clinical Feature Engineering**: Implements an `ICD9GroupTransformer` that programmatically groups granular ICD-9 codes in `diag_1`, `diag_2`, and `diag_3` into 9 clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Neoplasms, Genitourinary, and Other).
4. **Probability Calibration**: Calibrates prediction confidence values using **Isotonic Regression** so risk probabilities accurately correspond to actual clinical readmission likelihoods, reporting Brier score improvements and generating calibration curves.
5. **Precision-Recall Threshold Optimization**: Selects an optimal decision threshold that prioritizes clinical safety (high recall/sensitivity) to ensure high-risk patients are caught, while maintaining a reasonable precision to avoid clinician alert fatigue.
6. **Explainability & Trust (SHAP)**: Computes global and local Shapley values to make predictions understandable for medical staff. Explains global importance drivers and details individual true positive, false positive, and false negative patient cases.
7. **Algorithmic Fairness (Bias Audit)**: Disaggregates Precision, Recall, Selection Rate, and FPR across demographic subgroups (`race`, `gender`, `age`) to audit for disparities, exporting results in `outputs/bias_audit.csv`.

---

## 📁 Repository Layout

```
Hospital_readmission_riskpredictor/
├── data/                         # Created automatically; holds raw data
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── notebooks/                    # Interactive analysis notebooks
│   ├── eda.ipynb                 # Ingestion, cleaning, target binarization, leakage, and basic EDA
│   └── modelling.ipynb           # Model training, calibration, tuning, SHAP, and bias audit
├── outputs/                      # Artifacts generated during training and auditing
│   ├── bias_audit.csv            # Detailed demographic subgroups fairness table
│   ├── calibration_curve.png     # Calibration curve before vs. after isotonic scaling
│   ├── precision_recall_curve.png# PR curve with the selected operating threshold
│   ├── shap_summary.png          # SHAP global feature importance plot
│   ├── shap_force_tp.png         # Local SHAP waterfall plot for a True Positive patient
│   ├── shap_force_fp.png         # Local SHAP waterfall plot for a False Positive patient
│   └── shap_force_fn.png         # Local SHAP waterfall plot for a False Negative patient
├── src/                          # Modular production python scripts
│   ├── download_data.py          # Programmatically downloads and unzips the UCI dataset
│   ├── icd9_transformer.py       # Custom Scikit-learn transformer for grouping ICD-9 codes
│   ├── pipeline_builder.py       # Builds the complete Scikit-learn preprocessor and pipeline
│   ├── train.py                  # Main modeling, calibration, evaluation, SHAP, and audit execution
│   ├── verify_pipeline.py        # Verification script to load and test pipeline.pkl
│   └── create_notebooks.py       # Script that programmatically builds the Jupyter Notebooks
├── .env.example                  # Environment configuration example
├── docker-compose.yml            # Docker orchestration configuration
├── Dockerfile                    # Container configuration file
├── MODEL_CARD.md                 # Technical specification of performance, calibration, SHAP, and fairness
├── pipeline.pkl                  # Serialized calibrated pipeline (created after training)
└── requirements.txt              # Pinned python library dependencies
```

---

## 🛠️ Installation & Getting Started

You can run this project locally or in a containerized environment using Docker.

### Option 1: Local Setup (Recommended)

Make sure you have **Python 3.12** installed on your system.

1. **Clone the repository** and navigate to its root folder.
2. **Create a virtual environment** and activate it:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the dataset**:
   ```bash
   python src/download_data.py
   ```
5. **Generate the Jupyter Notebooks**:
   ```bash
   python src/create_notebooks.py
   ```
6. **Execute the complete training, calibration, and auditing pipeline**:
   ```bash
   python src/train.py
   ```
7. **Verify the serialized pipeline file**:
   ```bash
   python src/verify_pipeline.py
   ```

To explore interactive analysis, run `jupyter notebook` in your terminal and open the files under `notebooks/`.

---

## 🐳 Option 2: Containerized Setup (Docker)

To run the pipeline and run a Jupyter Notebook instance in a pre-configured Docker container:

1. **Build and start the container**:
   ```bash
   docker-compose up --build
   ```
2. Open your web browser and go to `http://localhost:8888` to access Jupyter.
3. To execute the python pipeline or verification scripts directly inside the active container:
   ```bash
   # Download data in container
   docker exec -it readmission_risk_predictor python src/download_data.py

   # Build notebooks in container
   docker exec -it readmission_risk_predictor python src/create_notebooks.py

   # Execute model training and pipeline calibration
   docker exec -it readmission_risk_predictor python src/train.py

   # Verify the serialized pipeline.pkl
   docker exec -it readmission_risk_predictor python src/verify_pipeline.py
   ```

---

## 📊 Summary of Model Performance

Detailed results, calibrations, and ethical analysis are located in [MODEL_CARD.md](MODEL_CARD.md).

- **Discrimination**: Calibrated XGBoost achieved a **ROC-AUC of 0.6802** and **PR-AUC of 0.2362**, outperforming the Logistic Regression baseline (ROC-AUC 0.6686, PR-AUC 0.2212).
- **Calibration**: Brier score improved from **0.2188** to **0.0955** (lower is better) using isotonic calibration, materially improving probability quality.
- **Fairness Audit**: Subgroup analysis shows recall remains in the mid-70% range for the main gender groups, with age-group recall varying from about **40%** to **87%** across smaller cohorts.
