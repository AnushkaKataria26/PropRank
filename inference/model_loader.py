import sqlite3
import joblib
import torch
import logging
from collections import namedtuple
from models.ranknet import RankNetMLP

logger = logging.getLogger(__name__)

ModelBundle = namedtuple('ModelBundle', [
    'model',
    'feature_version_id',
    'vectorizer',
    'scaler',
    'encoder',
    'model_version_id'
])

def load_active_model(config):
    """
    Loads the currently active trained RankNetMLP model and its associated feature pipeline artifacts.
    """
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Get the single active model version
    cursor.execute("SELECT id, artifact_path, feature_version_id FROM model_versions WHERE is_active = 1")
    rows = cursor.fetchall()
    
    if len(rows) == 0:
        conn.close()
        raise ValueError("No active model found in model_versions. Ensure a model has been trained and marked active.")
    if len(rows) > 1:
        conn.close()
        raise RuntimeError(f"Database corruption: Found {len(rows)} active models, but exactly 1 is expected.")
        
    model_row = rows[0]
    model_version_id = model_row['id']
    artifact_path = model_row['artifact_path']
    feature_version_id = model_row['feature_version_id']
    
    # 2. Get the associated feature artifacts
    cursor.execute("SELECT tfidf_artifact_path, scaler_artifact_path FROM feature_versions WHERE id = ?", (feature_version_id,))
    feat_rows = cursor.fetchall()
    conn.close()
    
    if len(feat_rows) == 0:
        raise ValueError(f"feature_version_id {feature_version_id} referenced by active model not found in feature_versions.")
        
    feat_row = feat_rows[0]
    tfidf_path = feat_row['tfidf_artifact_path']
    scaler_path = feat_row['scaler_artifact_path']
    
    # 3. Load feature artifacts
    try:
        vectorizer = joblib.load(tfidf_path)
        scaler, encoder = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load feature artifacts: {e}")
        
    # 4. Load the model state and infer input dimension
    try:
        state_dict = torch.load(artifact_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint from {artifact_path}: {e}")
        
    if 'mlp.0.weight' not in state_dict:
        raise ValueError("Model checkpoint does not contain expected keys (e.g., 'mlp.0.weight').")
        
    input_dim = state_dict['mlp.0.weight'].shape[1]
    
    # Instantiate model
    model = RankNetMLP(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()  # Important: disable dropout during inference
    
    logger.info(f"Loaded active model_version_id={model_version_id} (feature_version_id={feature_version_id})")
    
    return ModelBundle(
        model=model,
        feature_version_id=feature_version_id,
        vectorizer=vectorizer,
        scaler=scaler,
        encoder=encoder,
        model_version_id=model_version_id
    )
