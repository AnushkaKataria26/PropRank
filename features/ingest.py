import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_items_csv(csv_path, numerical_columns, categorical_columns):
    """
    Reads a CSV file containing items and validates it.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {csv_path}: {e}")
        
    if 'item_id' not in df.columns:
        raise ValueError("Missing required column: 'item_id'")
        
    if df['item_id'].isnull().any():
        raise ValueError("'item_id' column contains null values")
        
    duplicates = df['item_id'][df['item_id'].duplicated()]
    if not duplicates.empty:
        raise ValueError(f"Duplicate 'item_id's found: {duplicates.tolist()}")
        
    if 'text_description' not in df.columns:
        logger.warning("'text_description' column missing. Filling with empty strings.")
        df['text_description'] = ""
    else:
        df['text_description'] = df['text_description'].fillna("")
        
    for col in numerical_columns:
        if col not in df.columns:
            raise ValueError(f"Missing numerical column: '{col}'")
            
    for col in categorical_columns:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: '{col}'")
            
    return df
