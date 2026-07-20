import sqlite3
import logging
import json
import numpy as np

from inference.model_loader import load_active_model
from models.evaluate import compute_ndcg_at_10

logger = logging.getLogger(__name__)

def should_retrain(config):
    """
    Evaluates whether the model should be retrained based on:
    Condition A: Number of new preference pairs accumulated since last training >= threshold.
    Condition B: NDCG@10 on the most recent 100 pairs < quality floor.
    """
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check for active model
    cursor.execute("SELECT id, trained_at, training_pair_count, feature_version_id FROM model_versions WHERE is_active = 1")
    rows = cursor.fetchall()
    
    if len(rows) == 0:
        conn.close()
        return {
            "should_retrain": True,
            "reason": "no_active_model",
            "current_pair_count": 0,
            "recent_ndcg": None
        }
        
    active_model = rows[0]
    trained_at = active_model["trained_at"]
    feature_version_id = active_model["feature_version_id"]
    
    # Condition A: Pair accumulation
    cursor.execute("SELECT COUNT(*) as count FROM preference_log WHERE timestamp > ?", (trained_at,))
    new_pairs_count = cursor.fetchone()["count"]
    
    cond_a_met = new_pairs_count >= config.retrain_pair_threshold
    
    # Fetch total pairs to check if we have enough for Condition B
    cursor.execute("SELECT COUNT(*) as count FROM preference_log")
    total_pairs = cursor.fetchone()["count"]
    
    if total_pairs < 100:
        conn.close()
        # Not enough pairs to meaningfully evaluate Condition B
        reason = "pair_threshold" if cond_a_met else None
        return {
            "should_retrain": cond_a_met,
            "reason": reason,
            "current_pair_count": new_pairs_count,
            "recent_ndcg": None
        }
        
    # Condition B: Quality degradation
    query = """
        SELECT 
            p.item_i_id, p.item_j_id, p.winner_id, p.query_context, p.position_i, p.position_j,
            i.feature_vector_json AS features_i,
            j.feature_vector_json AS features_j
        FROM preference_log p
        LEFT JOIN items i ON p.item_i_id = i.item_id AND i.feature_version_id = ?
        LEFT JOIN items j ON p.item_j_id = j.item_id AND j.feature_version_id = ?
        WHERE p.source IN ('simulated', 'real')
        ORDER BY p.timestamp DESC
        LIMIT 100
    """
    
    cursor.execute(query, (feature_version_id, feature_version_id))
    recent_rows = cursor.fetchall()
    conn.close()
    
    valid_pairs = []
    expected_dim = None
    
    for row in recent_rows:
        features_i_str = row['features_i']
        features_j_str = row['features_j']
        
        if features_i_str is None or features_j_str is None:
            continue
            
        try:
            feat_i = np.array(json.loads(features_i_str), dtype=np.float32)
            feat_j = np.array(json.loads(features_j_str), dtype=np.float32)
        except Exception:
            continue
            
        if expected_dim is None:
            expected_dim = feat_i.shape[0]
            
        if feat_i.shape[0] != expected_dim or feat_j.shape[0] != expected_dim:
            continue
            
        label = 1.0 if row['winner_id'] == row['item_i_id'] else 0.0
            
        valid_pairs.append({
            'item_i_id': row['item_i_id'],
            'item_j_id': row['item_j_id'],
            'item_i_features': feat_i,
            'item_j_features': feat_j,
            'label': label,
            'query_context': row['query_context'],
            'position_i': row['position_i'],
            'position_j': row['position_j']
        })
        
    recent_ndcg = None
    cond_b_met = False
    
    if len(valid_pairs) > 0:
        # Load the active model to compute NDCG
        try:
            bundle = load_active_model(config)
            recent_ndcg = compute_ndcg_at_10(bundle.model, valid_pairs)
            cond_b_met = recent_ndcg < config.ndcg_retrain_floor
        except Exception as e:
            logger.error(f"Failed to evaluate recent pairs for Condition B: {e}")
            
    reason = None
    if cond_a_met and cond_b_met:
        reason = "both"
    elif cond_a_met:
        reason = "pair_threshold"
    elif cond_b_met:
        reason = "quality_floor"
        
    return {
        "should_retrain": cond_a_met or cond_b_met,
        "reason": reason,
        "current_pair_count": new_pairs_count,
        "recent_ndcg": recent_ndcg
    }
