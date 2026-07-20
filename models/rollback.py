import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

def rollback_to_version(model_version_id, config):
    """
    Rolls back the active model to the specified model_version_id.
    """
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Validate the model_version_id exists
    cursor.execute("SELECT * FROM model_versions WHERE id = ?", (model_version_id,))
    row = cursor.fetchone()
    
    if row is None:
        conn.close()
        raise ValueError(f"Model version ID {model_version_id} does not exist in model_versions table.")
        
    # 2. Validate the artifact_path exists on disk
    artifact_path = row["artifact_path"]
    if not os.path.exists(artifact_path):
        conn.close()
        raise FileNotFoundError(
            f"Cannot rollback to model version {model_version_id}. "
            f"The artifact file is missing from disk: {artifact_path}"
        )
        
    # 3. Single transaction to update is_active
    try:
        cursor.execute("BEGIN TRANSACTION")
        cursor.execute("UPDATE model_versions SET is_active = 0")
        cursor.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (model_version_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        raise RuntimeError(f"Failed to update active model version: {e}")
        
    conn.close()
    
    logger.info(f"Successfully rolled back to model version {model_version_id}")
    
    return {
        "model_version_id": model_version_id,
        "artifact_path": artifact_path,
        "ndcg_at_10": row["ndcg_at_10"],
        "pairwise_accuracy": row["pairwise_accuracy"],
        "map_score": row["map_score"],
        "is_bias_corrected": row["is_bias_corrected"]
    }
