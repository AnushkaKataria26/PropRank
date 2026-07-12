import logging
from config.loader import get_config
from features.ingest import load_items_csv
from features.build_features import build_feature_vectors
from features.persist import save_feature_version
from features.store import persist_item_features

class WarningCatcher(logging.Handler):
    def __init__(self):
        super().__init__()
        self.warnings = []
        
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.warnings.append(self.format(record))

def run_feature_pipeline(csv_path, numerical_columns, categorical_columns):
    config = get_config()
    
    logger = logging.getLogger('features')
    logger.setLevel(logging.WARNING)
    catcher = WarningCatcher()
    logger.addHandler(catcher)
    
    print(f"Starting feature pipeline for {csv_path}...")
    
    # 1. Ingest
    df = load_items_csv(csv_path, numerical_columns, categorical_columns)
    print(f"Ingested {len(df)} items.")
    
    # 2. Build Features
    feature_matrix, vectorizer, scaler, encoder = build_feature_vectors(
        df, numerical_columns, categorical_columns, config
    )
    print(f"Built feature vectors. Dimensionality: {feature_matrix.shape[1]}")
    
    # 3. Persist Artifacts
    feature_version_id = save_feature_version(vectorizer, scaler, encoder, config)
    print(f"Saved feature version ID: {feature_version_id}")
    
    # 4. Store in SQLite
    persist_item_features(df, feature_matrix, feature_version_id, config)
    print("Successfully wrote features to SQLite.")
    
    logger.removeHandler(catcher)
    
    if catcher.warnings:
        print("\n--- Pipeline Warnings ---")
        for w in catcher.warnings:
            print(f"- {w}")
            
    return feature_version_id
