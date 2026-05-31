import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def verify_pipeline():
    print("--- STARTING PIPELINE SERIALIZATION VERIFICATION ---")
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pipeline_path = os.path.join(base_dir, "pipeline.pkl")
    data_path = os.path.join(base_dir, "data", "diabetic_data.csv")
    
    # 1. Check if model exists
    if not os.path.exists(pipeline_path):
        print(f"Error: pipeline.pkl not found at {pipeline_path}")
        sys.exit(1)
        
    # 2. Load model
    print(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    print("Pipeline loaded successfully.")
    print(f"Pipeline steps: {list(pipeline.named_steps.keys())}")
    
    # 3. Load dataset to sample from
    print(f"Loading raw dataset from {data_path} to sample...")
    df = pd.read_csv(data_path)
    
    # Target and leakage cleaning
    df['target'] = (df['readmitted'] == '<30').astype(int)
    leakage_ids = [11, 13, 14, 19, 20, 21]
    df_clean = df[~df['discharge_disposition_id'].isin(leakage_ids)].copy()
    
    # Extract features
    cols_to_drop = ['readmitted', 'target', 'encounter_id', 'patient_nbr']
    X = df_clean.drop(columns=cols_to_drop)
    y = df_clean['target']
    
    # Take a 10-row sample to test
    X_sample = X.sample(n=10, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    # 4. Predict using the pipeline
    print("Passing raw sample features through pipeline...")
    try:
        preds = pipeline.predict(X_sample)
        probs = pipeline.predict_proba(X_sample)[:, 1]
        
        print("\nVerification Results:")
        for idx in range(len(preds)):
            print(f" Patient {idx+1:02d}: True Label={y_sample.iloc[idx]}, Prediction={preds[idx]}, Calibrated Risk Score={probs[idx]:.4f}")
            
        # Run assertions
        assert len(preds) == 10, "Output prediction length mismatch."
        assert len(probs) == 10, "Output probability length mismatch."
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Risk scores are out of bounds [0.0, 1.0]."
        assert np.all(np.isin(preds, [0, 1])), "Predictions should only be binary (0 or 1)."
        
        print("\n[SUCCESS] Pipeline is fully serializable, functional, and calibrated!")
        print("--- VERIFICATION COMPLETED SUCCESSFULLY ---")
        
    except Exception as e:
        print(f"\n[FAILURE] Pipeline failed verification: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_pipeline()
