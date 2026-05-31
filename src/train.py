import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import xgboost as xgb
import shap

# Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.pipeline_builder import build_full_pipeline

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def main():
    print("--- STARTING ML PIPELINE EXECUTION ---")
    
    # 1. Paths & Directory Setup
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "diabetic_data.csv")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 2. Data Load & Cleaning
    print("\n[Step 1] Loading and cleaning dataset...")
    df = pd.read_csv(data_path)
    print(f"Original shape: {df.shape}")
    
    # Binarize readmitted target variable
    # Positive class (1): readmitted <30 days
    # Negative class (0): readmitted >30 days or NO readmission
    df['target'] = (df['readmitted'] == '<30').astype(int)
    print(f"Target distribution (binarized):")
    print(df['target'].value_counts(normalize=True))
    
    # Remove Data Leakage rows
    # discharge_disposition_id values representing deceased or hospice:
    # 11: Expired
    # 13: Hospice / home
    # 14: Hospice / medical facility
    # 19: Expired at home (Medicaid only, hospice)
    # 20: Expired in a medical facility (Medicaid only, hospice)
    # 21: Expired, place unknown (Medicaid only, hospice)
    leakage_ids = [11, 13, 14, 19, 20, 21]
    df_clean = df[~df['discharge_disposition_id'].isin(leakage_ids)].copy()
    print(f"Shape after removing data leakage rows (expired/hospice): {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} rows due to leakage.")
    
    # 3. Train/Test Split
    print("\n[Step 2] Splitting dataset...")
    # Drop target columns, encounter/patient IDs to prevent target leakage
    cols_to_drop = ['readmitted', 'target', 'encounter_id', 'patient_nbr']
    X = df_clean.drop(columns=cols_to_drop)
    y = df_clean['target']
    
    # Stratified split because of high class imbalance (~11% readmissions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    print(f"Train readmission rate: {y_train.mean():.4f}, Test readmission rate: {y_test.mean():.4f}")
    
    # 4. Baseline Model: Logistic Regression
    print("\n[Step 3] Training baseline Logistic Regression...")
    lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_pipeline = build_full_pipeline(lr_clf)
    
    print("Fitting baseline pipeline...")
    lr_pipeline.fit(X_train, y_train)
    
    # Evaluate baseline
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, lr_probs)
    lr_pr_auc = auc(recall_lr, precision_lr)
    
    print(f"Baseline Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print(f"Baseline Logistic Regression PR-AUC: {lr_pr_auc:.4f}")
    
    # 5. Primary Model: XGBoost & Hyperparameter Optimization
    print("\n[Step 4] Training primary XGBoost classifier...")
    # Calculate scale_pos_weight to handle class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos = num_neg / num_pos
    print(f"XGBoost scale_pos_weight for imbalance: {scale_pos:.4f}")
    
    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos,
        random_state=42
    )
    
    xgb_pipeline = build_full_pipeline(xgb_clf)
    
    # Hyperparameter tuning parameters
    param_dist = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [4, 5, 6],
        'classifier__learning_rate': [0.03, 0.05, 0.1],
        'classifier__subsample': [0.8, 0.9],
        'classifier__colsample_bytree': [0.8, 0.9]
    }
    
    print("Running RandomizedSearchCV...")
    # Use 3-fold cross-validation and 6 search iterations to be efficient yet robust
    random_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    
    best_xgb_pipeline = random_search.best_estimator_
    print(f"Best hyperparameters found:")
    for param_name, param_val in random_search.best_params_.items():
        print(f" - {param_name}: {param_val}")
        
    xgb_raw_probs = best_xgb_pipeline.predict_proba(X_test)[:, 1]
    xgb_raw_auc = roc_auc_score(y_test, xgb_raw_probs)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_raw_probs)
    xgb_raw_pr_auc = auc(recall_xgb, precision_xgb)
    
    print(f"Tuned XGBoost (Raw Probs) ROC-AUC: {xgb_raw_auc:.4f}")
    print(f"Tuned XGBoost (Raw Probs) PR-AUC: {xgb_raw_pr_auc:.4f}")
    
    # 6. Model Calibration
    print("\n[Step 5] Performing probability calibration...")
    # We will calibrate the best XGBoost classifier using isotonic calibration
    # Wrap the classifier part of the pipeline and keep preprocessing intact
    preprocessor = best_xgb_pipeline.named_steps['preprocessor']
    best_xgb_clf = best_xgb_pipeline.named_steps['classifier']
    
    # We calibrate the classifier using 5-fold cross-validation
    calibrated_clf = CalibratedClassifierCV(
        estimator=best_xgb_clf,
        method='isotonic',
        cv=5
    )
    
    # Assemble the final calibrated pipeline
    calibrated_pipeline = build_full_pipeline(calibrated_clf)
    print("Fitting calibrated pipeline...")
    calibrated_pipeline.fit(X_train, y_train)
    
    # Save the calibrated pipeline early in case of later script hiccups
    pipeline_save_path = os.path.join(base_dir, "pipeline.pkl")
    joblib.dump(calibrated_pipeline, pipeline_save_path)
    print(f"Calibrated pipeline saved to: {pipeline_save_path}")
    
    # Evaluate calibrated probabilities
    cal_probs = calibrated_pipeline.predict_proba(X_test)[:, 1]
    cal_auc = roc_auc_score(y_test, cal_probs)
    precision_cal, recall_cal, _ = precision_recall_curve(y_test, cal_probs)
    cal_pr_auc = auc(recall_cal, precision_cal)
    
    print(f"Calibrated XGBoost ROC-AUC: {cal_auc:.4f}")
    print(f"Calibrated XGBoost PR-AUC: {cal_pr_auc:.4f}")
    
    # Calculate Brier Scores
    brier_raw = brier_score_loss(y_test, xgb_raw_probs)
    brier_cal = brier_score_loss(y_test, cal_probs)
    print(f"Brier score (Raw XGBoost): {brier_raw:.4f}")
    print(f"Brier score (Calibrated XGBoost): {brier_cal:.4f}")
    
    # Save Calibration Curve Plot
    print("Plotting and saving calibration curves...")
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, xgb_raw_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#d95f02", label=f"Raw XGBoost (Brier: {brier_raw:.4f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#1b9e77", label=f"Calibrated XGBoost (Brier: {brier_cal:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Probability Calibration Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    cal_plot_path = os.path.join(outputs_dir, "calibration_curve.png")
    plt.savefig(cal_plot_path)
    plt.close()
    print(f"Calibration curve plot saved to: {cal_plot_path}")
    
    # 7. Threshold Optimization
    print("\n[Step 6] Optimizing decision threshold using PR Curve...")
    # Find operating threshold prioritizing Recall while keeping Precision reasonable
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, cal_probs)
    
    # Select threshold where Recall is around 75-80% to catch readmissions
    target_recall = 0.75
    # Find threshold index closest to target recall
    idx = np.argmin(np.abs(recall_curve - target_recall))
    selected_threshold = thresholds[idx]
    
    # Let's verify metrics at this selected threshold
    test_preds_selected = (cal_probs >= selected_threshold).astype(int)
    sel_precision = precision_score(y_test, test_preds_selected)
    sel_recall = recall_score(y_test, test_preds_selected)
    sel_f1 = f1_score(y_test, test_preds_selected)
    
    print(f"Selected Decision Threshold: {selected_threshold:.4f}")
    print(f" - Precision at threshold: {sel_precision:.4f}")
    print(f" - Recall at threshold: {sel_recall:.4f}")
    print(f" - F1-Score at threshold: {sel_f1:.4f}")
    
    # Plot PR Curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(recall_curve, precision_curve, color="#2b5c8f", lw=2, label=f"PR Curve (AUC = {cal_pr_auc:.4f})")
    ax.plot(sel_recall, sel_precision, "ro", markersize=8, label=f"Selected Threshold ({selected_threshold:.2f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall Curve with Selected Operating Threshold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    pr_plot_path = os.path.join(outputs_dir, "precision_recall_curve.png")
    plt.savefig(pr_plot_path)
    plt.close()
    print(f"PR curve plot saved to: {pr_plot_path}")
    
    # 8. Explainability: SHAP Values
    print("\n[Step 7] Generating SHAP explanations...")
    # Extract fit preprocessor to transform test set for SHAP
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    if sparse.issparse(X_train_preprocessed):
        X_train_preprocessed = X_train_preprocessed.toarray()
    if sparse.issparse(X_test_preprocessed):
        X_test_preprocessed = X_test_preprocessed.toarray()
    
    # Get preprocessed feature names out
    feature_names = preprocessor.get_feature_names_out()
    
    # Clean feature names (remove prefix num__, cat__, diag__ for readability in plots)
    clean_feature_names = []
    for name in feature_names:
        clean_name = name
        for prefix in ['num__', 'cat__', 'diag__']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        clean_name = clean_name.replace("?", "Missing")
        clean_name = re.sub(r"[^0-9A-Za-z_\.\-]+", "_", clean_name)
        clean_name = clean_name.strip("_") or "feature"
        clean_feature_names.append(clean_name)
        
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=clean_feature_names)
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=clean_feature_names)
    
    # Compute SHAP values on preprocessed test set using TreeExplainer
    # TreeExplainer is used on the underlying XGBoost estimator
    explainer = shap.TreeExplainer(best_xgb_clf)
    print("Computing SHAP values for test set...")
    # Using a subset of test set (e.g. 500 samples) to ensure SHAP runs extremely fast
    shap_sample_size = min(500, X_test_preprocessed_df.shape[0])
    X_test_shap_subset = X_test_preprocessed_df.iloc[:shap_sample_size]
    shap_values = explainer(X_test_shap_subset)
    
    # Save SHAP Global Summary Plot
    plt.figure(figsize=(10, 6), dpi=150)
    shap.summary_plot(shap_values, X_test_shap_subset, show=False)
    plt.title("SHAP Global Feature Importance Summary", fontsize=14, pad=15)
    plt.tight_layout()
    shap_summary_path = os.path.join(outputs_dir, "shap_summary.png")
    plt.savefig(shap_summary_path)
    plt.close()
    print(f"SHAP global summary plot saved to: {shap_summary_path}")
    
    # Generate local force plots for at least three specific patient predictions:
    # 1. True Positive (predicted risk >= selected_threshold, y_test == 1)
    # 2. False Positive (predicted risk >= selected_threshold, y_test == 0)
    # 3. False Negative (predicted risk < selected_threshold, y_test == 1)
    print("Generating local SHAP force plots...")
    
    # We search the whole test set for these examples
    tp_idx, fp_idx, fn_idx = None, None, None
    for i in range(len(y_test)):
        prob = cal_probs[i]
        true_label = y_test.iloc[i]
        
        if tp_idx is None and prob >= selected_threshold and true_label == 1:
            tp_idx = i
        elif fp_idx is None and prob >= selected_threshold and true_label == 0:
            fp_idx = i
        elif fn_idx is None and prob < selected_threshold and true_label == 1:
            fn_idx = i
            
        if tp_idx is not None and fp_idx is not None and fn_idx is not None:
            break
            
    print(f"Selected Local Patient Indices: TP={tp_idx}, FP={fp_idx}, FN={fn_idx}")
    
    # Helper to generate and save a force plot as matplotlib
    # To save a force plot in SHAP, we pass matplotlib=True to force_plot
    def save_force_plot(patient_idx, label, filename):
        if patient_idx is None:
            print(f"Warning: Could not find a patient with label type {label}")
            return
            
        # Re-explain the single patient instance
        patient_data = X_test_preprocessed_df.iloc[[patient_idx]]
        patient_shap = explainer(patient_data)
        
        plt.figure(figsize=(12, 4), dpi=150)
        shap.plots.waterfall(patient_shap[0], show=False)
        plt.title(f"SHAP Local Waterfall Plot: {label} Patient (Risk: {cal_probs[patient_idx]:.2%})", fontsize=14, pad=15)
        plt.tight_layout()
        save_path = os.path.join(outputs_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Local {label} plot saved to: {save_path}")
        
    save_force_plot(tp_idx, "True Positive", "shap_force_tp.png")
    save_force_plot(fp_idx, "False Positive", "shap_force_fp.png")
    save_force_plot(fn_idx, "False Negative", "shap_force_fn.png")
    
    # 9. Bias Audit
    print("\n[Step 8] Performing bias and fairness audit...")
    # Disaggregate Recall, Precision, Selection Rate, False Positive Rate (FPR)
    # by race, gender, and age
    audit_results = []
    
    # Subgroups list
    subgroup_features = ['race', 'gender', 'age']
    
    # Pre-calculate predictions at selected threshold for the entire test set
    test_preds = (cal_probs >= selected_threshold).astype(int)
    
    for feat in subgroup_features:
        # Get unique values in X_test, handling missing "?" or empty string
        unique_vals = X_test[feat].unique()
        
        for val in unique_vals:
            # Subset indices
            mask = X_test[feat] == val
            sub_X = X_test[mask]
            sub_y = y_test[mask]
            sub_preds = test_preds[mask]
            sub_probs = cal_probs[mask]
            
            n_samples = len(sub_y)
            if n_samples < 30: # Exclude tiny subgroups for statistical stability
                continue
                
            n_pos = sub_y.sum()
            n_neg = n_samples - n_pos
            
            # Selection Rate
            sel_rate = sub_preds.mean()
            
            # Precision
            if sub_preds.sum() > 0:
                prec = precision_score(sub_y, sub_preds, zero_division=0)
            else:
                prec = 0.0
                
            # Recall
            if n_pos > 0:
                rec = recall_score(sub_y, sub_preds, zero_division=0)
            else:
                rec = 0.0
                
            # False Positive Rate (FPR) = FP / N
            # FP is when pred == 1 and true == 0
            if n_neg > 0:
                fp = ((sub_preds == 1) & (sub_y == 0)).sum()
                fpr = fp / n_neg
            else:
                fpr = 0.0
                
            # AUC-ROC for subgroup
            if n_pos > 0 and n_neg > 0:
                sub_auc = roc_auc_score(sub_y, sub_probs)
            else:
                sub_auc = np.nan
                
            audit_results.append({
                'Feature': feat,
                'Subgroup': str(val),
                'Sample Size': n_samples,
                'Readmission Count': n_pos,
                'Readmission Rate': n_pos / n_samples,
                'Selection Rate': sel_rate,
                'Precision': prec,
                'Recall': rec,
                'False Positive Rate': fpr,
                'AUC-ROC': sub_auc
            })
            
    audit_df = pd.DataFrame(audit_results)
    
    # Save bias audit report
    audit_csv_path = os.path.join(outputs_dir, "bias_audit.csv")
    audit_df.to_csv(audit_csv_path, index=False)
    print(f"Bias audit results successfully saved to: {audit_csv_path}")
    
    # Print out summary table
    print("\n--- BIAS AUDIT SUMMARY TABLE ---")
    print(audit_df[['Feature', 'Subgroup', 'Sample Size', 'Readmission Rate', 'Selection Rate', 'Precision', 'Recall', 'AUC-ROC']].to_string(index=False))
    
    print("\n--- ML PIPELINE EXECUTION COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()
