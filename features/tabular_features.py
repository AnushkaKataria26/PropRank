import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

logger = logging.getLogger(__name__)

def fit_numerical(df, numerical_columns, fillna_strategy='raise'):
    if not numerical_columns:
        return None, np.empty((len(df), 0))
        
    num_data = df[numerical_columns].copy()
    
    if num_data.isnull().any().any():
        if fillna_strategy == 'raise':
            cols_with_nan = num_data.columns[num_data.isnull().any()].tolist()
            raise ValueError(f"NaN found in numerical columns: {cols_with_nan}")
        elif fillna_strategy == 'mean':
            num_data = num_data.fillna(num_data.mean())
        elif fillna_strategy == 'zero':
            num_data = num_data.fillna(0)
        else:
            raise ValueError(f"Unknown fillna_strategy: {fillna_strategy}")
            
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(num_data)
    return scaler, scaled_data

def fit_categorical(df, categorical_columns):
    if not categorical_columns:
        return None, np.empty((len(df), 0))
        
    cat_data = df[categorical_columns].copy()
    
    for col in categorical_columns:
        nunique = cat_data[col].nunique(dropna=False)
        if nunique > 50:
            logger.warning(f"Categorical column '{col}' has high cardinality ({nunique}). One-hot encoding will be wide.")
            
    # handle_unknown='ignore' produces an all-zero encoding for unseen categories
    # Convert to string to avoid mixed type issues with unseen categories.
    cat_data = cat_data.astype(str)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(cat_data)
    return encoder, encoded_data
