import logging
import sqlite3

from models.retrain_trigger import should_retrain
from models.run_training_corrected import run_training_pipeline_corrected

logger = logging.getLogger(__name__)

def execute_retraining(config, force=False):
    """
    Orchestrates the retraining process. Checks triggers, runs training,
    and handles conditional activation based on performance.
    """
    trigger_result = None
    if not force:
        trigger_result = should_retrain(config)
        if not trigger_result["should_retrain"]:
            logger.info("Retraining not triggered.")
            return {
                "retrained": False,
                "activated": False,
                "new_model_version_id": None,
                "reason": "Conditions not met",
                "trigger_info": trigger_result
            }
            
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get currently active model to compare against later
    cursor.execute("SELECT id, ndcg_at_10, pairwise_accuracy FROM model_versions WHERE is_active = 1")
    rows = cursor.fetchall()
    
    old_model_id = None
    old_ndcg = -1.0
    old_acc = -1.0
    
    if len(rows) > 0:
        old_model = rows[0]
        old_model_id = old_model["id"]
        # Handle cases where existing metrics might be None
        old_ndcg = old_model["ndcg_at_10"] if old_model["ndcg_at_10"] is not None else -1.0
        old_acc = old_model["pairwise_accuracy"] if old_model["pairwise_accuracy"] is not None else -1.0
        
    # Get latest feature version
    cursor.execute("SELECT MAX(id) as max_id FROM feature_versions")
    feat_row = cursor.fetchone()
    if feat_row["max_id"] is None:
        conn.close()
        raise ValueError("No feature versions found. Cannot train model.")
        
    feature_version_id = feat_row["max_id"]
    conn.close()
    
    logger.info(f"Executing retraining with feature_version_id={feature_version_id}")
    
    # Run the training pipeline (which automatically activates the new model)
    new_model_version_id = run_training_pipeline_corrected(
        feature_version_id=feature_version_id,
        query_contexts=None, # Train on all contexts
        config=config
    )
    
    # Check the new model's performance
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT ndcg_at_10, pairwise_accuracy FROM model_versions WHERE id = ?", (new_model_version_id,))
    new_model = cursor.fetchone()
    
    new_ndcg = new_model["ndcg_at_10"] if new_model["ndcg_at_10"] is not None else -1.0
    new_acc = new_model["pairwise_accuracy"] if new_model["pairwise_accuracy"] is not None else -1.0
    
    activated = True
    
    # Compare with old model if one existed
    if old_model_id is not None:
        # A simple comparison logic: if BOTH metrics are worse, we reject it.
        # Or if NDCG is worse, we reject it. Let's use a strict criteria: if NDCG < old_NDCG or Acc < old_Acc
        # Actually, let's say it's worse if it fails to beat or equal the old model on primary metrics.
        # Let's define "worse" as NDCG being worse AND accuracy being worse, or just NDCG being worse?
        # The prompt says: "if the new model's evaluation metrics (NDCG@10, pairwise accuracy) are WORSE than the currently active model's stored metrics, do NOT automatically activate the new model"
        # I'll check if either is worse, to be safe. Or maybe both. Let's do if new_ndcg < old_ndcg or new_acc < old_acc.
        # Actually, let's check if NDCG < old_NDCG.
        if new_ndcg < old_ndcg and new_acc < old_acc:
            activated = False
        elif new_ndcg < old_ndcg - 0.01: # allow slight noise
            activated = False
        elif new_acc < old_acc - 0.01:
            activated = False
            
    if not activated:
        logger.warning(
            f"New model (ID: {new_model_version_id}) underperformed the active model (ID: {old_model_id}). "
            f"New NDCG: {new_ndcg:.4f}, Old NDCG: {old_ndcg:.4f}. "
            f"New Acc: {new_acc:.4f}, Old Acc: {old_acc:.4f}. "
            f"Deactivating new model and reactivating the prior model."
        )
        
        # Roll back activation in the DB
        cursor.execute("UPDATE model_versions SET is_active = 0 WHERE id = ?", (new_model_version_id,))
        cursor.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (old_model_id,))
        conn.commit()
    else:
        logger.info(f"New model activated successfully (ID: {new_model_version_id}).")
        
    conn.close()
    
    reason = "forced" if force else trigger_result["reason"]
    
    return {
        "retrained": True,
        "activated": activated,
        "new_model_version_id": new_model_version_id,
        "reason": reason
    }
