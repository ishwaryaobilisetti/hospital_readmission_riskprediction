# Model Card - 30-Day Hospital Readmission Risk Predictor

This model card details the design, performance, calibration, explainability, and fairness auditing of the calibrated machine learning system built to predict 30-day hospital readmissions for diabetic patients.

---

## 1. Model Details

- **Developer**: Antigravity AI Coding Assistant & Clinical Data Team
- **Model Date**: May 2026
- **Model Version**: v1.0.0
- **Model Type**: Calibrated Extreme Gradient Boosting (XGBoost) Classifier Pipeline
- **Dependencies**: `scikit-learn`, `xgboost`, `shap`, `pandas`, `numpy`, `matplotlib`
- **Citation**: Based on Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database."

---

## 2. Intended Use

- **Intended Users**: Clinical staff, discharge coordinators, and hospital transitional care teams.
- **Intended Applications**: Used at the point of discharge to identify patients at high risk of 30-day readmission. Allows clinical staff to schedule timely follow-up appointments, perform medication reconciliations, and assign home health care services.
- **Out of Scope**:
  - Automated refusal of discharge or billing decisions.
  - Direct medical treatment choices without clinical oversight.
  - Non-diabetic patient encounters (model is trained and optimized specifically for diabetic clinical care profiles).

---

## 3. Training & Preprocessing Data

- **Dataset Source**: Diabetes 130-US Hospitals (1999–2008) dataset, UCI Machine Learning Repository.
- **Size**: 101,766 raw hospital encounters across a decade.
- **Leakage Filtering**: Excluded 2,423 rows where the patient expired (died) or was discharged to hospice (discharge disposition IDs: 11, 13, 14, 19, 20, 21), since these patients could not be readmitted.
- **Preprocessing Pipeline**:
  - Encapsulated in a single serializable Scikit-learn `Pipeline`.
  - **Numeric columns**: Median imputed and scaled with `StandardScaler`.
  - **Categorical columns**: Replaced missing values (`?`) with `'Missing'` to maintain representation and avoid bias, followed by `OneHotEncoder(handle_unknown='ignore')`.
  - **ICD-9 diagnosis columns (`diag_1`, `diag_2`, `diag_3`)**: Passed through a custom `ICD9GroupTransformer` mapping granular codes to 9 clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Neoplasms, Genitourinary, and Other) followed by one-hot encoding.
  - **Target Variable**: Binarized (`<30` readmission -> 1; `>30` and `NO` -> 0).

---

## 4. Evaluation Metrics & Performance

The model was evaluated on a stratified 20% test split of the non-leaky dataset.

### 4.1. Overall Performance Summary
- **Baseline Logistic Regression**:
  - ROC-AUC: 0.6686
  - PR-AUC: 0.2212
- **Calibrated XGBoost Classifier**:
  - ROC-AUC: 0.6802
  - PR-AUC: 0.2362
  - *Note: XGBoost outperforms the Logistic Regression baseline in discrimination, though the lift is modest.*

### 4.2. Probability Calibration
Raw outputs from tree ensembles are often uncalibrated and overconfident. Using **Isotonic Regression** via `CalibratedClassifierCV(cv=5)` improves risk probability accuracy:
- **Brier Score (Raw XGBoost)**: 0.2188
- **Brier Score (Calibrated XGBoost)**: 0.0955 (lower is better, indicating better probability calibration)
- *The calibrated probabilities are substantially closer to observed outcomes, making risk scores more clinically interpretable.*

### 4.3. Threshold Optimization
In a clinical setting, missing a high-risk patient (False Negative) is much more costly than checking a low-risk patient (False Positive). Thus, we selected an operating threshold of **0.0904**, prioritizing **Recall**:
- **Operating Threshold**: 0.0904
- **Precision at threshold**: 0.1594
- **Recall at threshold**: 0.7499 (catches about 75% of readmissions)
- **F1-Score at threshold**: 0.2630

---

## 5. Explainability (SHAP Analysis)

To make model predictions trustworthy for clinicians, global and local feature importance was computed using SHAP values:

### 5.1. Global Feature Drivers
The top features pushing patients toward high readmission risk are:
1. **`number_inpatient`**: Previous inpatient visits in the past year is the single strongest predictor. Repeated hospitalizations indicate high clinical vulnerability.
2. **`number_emergency` / `number_outpatient`**: Prior utilization of emergency or outpatient care.
3. **`number_diagnoses`**: A high count of recorded diagnoses indicates patient comorbidity.
4. **`time_in_hospital`**: Length of stay of the current visit.
5. **`num_medications`**: High count of distinct medications.

### 5.2. Local Clinical Cases (Waterfall Explanations)
Three specific patient cases were extracted from the test set to explain local reasoning:
- **True Positive (TP)**: The model predicted a risk score of **24.5%** (well above our operating threshold). The main drivers pushing the risk up were a high number of prior inpatient visits (e.g., `number_inpatient=2`) and high comorbidity count (`number_diagnoses=9`). The patient was indeed readmitted within 30 days.
- **False Positive (FP)**: The model flagged a patient with a risk of **18.9%** (above threshold). The main risk factors were a long hospital stay (`time_in_hospital=7` days) and high medication count (`num_medications=28`). However, the patient was not readmitted. This patient represents a safe and reasonable clinical review (the "false alarm" is clinically justified given the complexity).
- **False Negative (FN)**: The model predicted a low risk of **7.5%** (below threshold). This patient had no prior inpatient visits (`number_inpatient=0`) and was relatively young. However, the patient was readmitted. This reveals a limitation of the model—missing readmissions driven by sudden acute complications rather than chronic instability.

---

## 6. Fairness & Bias Audit

A bias audit was performed by calculating disaggregated metrics across demographic groups:

### 6.1. Audit Results Table
Disaggregated metrics can be reviewed in detail in [outputs/bias_audit.csv](outputs/bias_audit.csv). A summary of performance across major groups:

| Feature | Subgroup | Sample Size | Readmission Rate | Selection Rate | Precision | Recall | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Race** | Caucasian | 14,743 | 11.6% | 54.7% | 16.1% | 75.9% | 0.681 |
| | African American | 3,809 | 11.0% | 52.9% | 15.2% | 73.0% | 0.669 |
| | Hispanic | 410 | 7.8% | 43.9% | 15.0% | 84.4% | 0.783 |
| | Asian | 128 | 15.6% | 44.5% | 29.8% | 85.0% | 0.807 |
| **Gender** | Female | 10,651 | 11.3% | 55.6% | 15.3% | 75.5% | 0.672 |
| | Male | 9,218 | 11.5% | 51.3% | 16.7% | 74.4% | 0.690 |
| **Age** | [70-80) | 5,027 | 11.6% | 60.6% | 15.0% | 78.5% | 0.670 |
| | [30-40) | 721 | 10.0% | 38.6% | 19.1% | 73.6% | 0.767 |

### 6.2. Fairness Key Findings
- **Disparity in Recall**: Recall varies across age cohorts, and smaller cohorts are noisier, so the gaps should be interpreted with sample size in mind.
- **Disparity in Race**: The minority groups are much smaller than the Caucasian cohort, so subgroup metrics are noisier and can swing substantially from one split to another.
- **Potential Hypotheses**:
  1. **Data Representation Imbalance**: The dataset is overwhelmingly composed of Caucasian and older patient cohorts. The model will naturally fit those majority groups more tightly.
  2. **Sampling Noise in Small Groups**: The smaller race and age cohorts can show large swings in precision/recall because a few predictions change the subgroup metric materially.

---

## 7. Ethical Considerations & Mitigations

1. **Alert Fatigue**: The low precision (~17%) means that for every 6 patients flagged, only 1 will actually readmit. While clinically safe (high recall), this could lead to alert fatigue. Hospitals should integrate the model as a supportive score rather than an intrusive alarm.
2. **Imputed Race & Demographics**: We avoided imputing missing values in the `race` feature to prevent masking performance gaps. In production, missing data must be carefully flagged to ensure equity of care.
3. **Data Antiquity**: The dataset is from 1999–2008. Medical practices and discharge protocols have shifted since. The model must be fine-tuned or recalibrated on modern hospital databases before real-world clinical deployment.
