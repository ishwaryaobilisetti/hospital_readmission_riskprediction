import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ICD9GroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns if columns is not None else ['diag_1', 'diag_2', 'diag_3']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        
        for col in self.columns:
            if col in X_out.columns:
                X_out[col] = X_out[col].apply(self._map_code)
        return X_out
        
    def _map_code(self, code):
        if pd.isna(code):
            return 'Other'
        code = str(code).strip()
        if not code or code == '?' or code == 'None' or code == '':
            return 'Other'
            
        # V or E codes
        if code.startswith('V') or code.startswith('E'):
            return 'Other'
            
        try:
            # Handle standard numeric codes
            # Many codes may have decimal points, e.g. 250.02
            # We can extract the integer part or try converting to float
            val = float(code)
            
            # Grouping ranges based on strack et al:
            # - Circulatory: 390-459, 785
            # - Respiratory: 460-519, 786
            # - Digestive: 520-579, 787
            # - Diabetes: 250.xx
            # - Injury: 800-999
            # - Musculoskeletal: 710-739
            # - Neoplasms: 140-239
            # - Genitourinary: 580-629, 788
            # - Other: All other codes (including V and E codes)
            
            if 250 <= val < 251:
                return 'Diabetes'
            elif (390 <= val <= 459) or val == 785:
                return 'Circulatory'
            elif (460 <= val <= 519) or val == 786:
                return 'Respiratory'
            elif (520 <= val <= 579) or val == 787:
                return 'Digestive'
            elif (580 <= val <= 629) or val == 788:
                return 'Genitourinary'
            elif 140 <= val <= 239:
                return 'Neoplasms'
            elif 710 <= val <= 739:
                return 'Musculoskeletal'
            elif 800 <= val <= 999:
                return 'Injury'
            else:
                return 'Other'
        except ValueError:
            return 'Other'
            
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.columns
        return input_features
