import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Add root directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.icd9_transformer import ICD9GroupTransformer


def cast_to_string(df):
    return df.astype(str)


class StringCaster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return cast_to_string(X)

    def get_feature_names_out(self, input_features=None):
        return input_features

def build_preprocessing_pipeline():
    # Numeric features to scale and impute
    numeric_features = [
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses'
    ]
    
    # Categorical features to encode
    categorical_features = [
        'race', 'gender', 'age', 'payer_code', 'medical_specialty',
        'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
        'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    ]
    
    # Diagnosis columns that need ICD-9 grouping first
    diag_features = ['diag_1', 'diag_2', 'diag_3']
    
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: replaces '?' (UCI missing value) with 'Missing' and OneHotEncodes
    # Note: IDs like admission_type_id are loaded as numeric by default, so we cast them to string first.
    categorical_transformer = Pipeline(steps=[
        ('string_caster', StringCaster()),
        ('imputer', SimpleImputer(missing_values='?', strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Diagnosis pipeline: applies ICD-9 grouping then OneHotEncodes
    diag_transformer = Pipeline(steps=[
        ('string_caster', StringCaster()),
        ('grouper', ICD9GroupTransformer(columns=diag_features)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('diag', diag_transformer, diag_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def build_full_pipeline(classifier):
    """
    Creates a full pipeline combining preprocessing and a given classifier.
    """
    preprocessor = build_preprocessing_pipeline()
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return full_pipeline
