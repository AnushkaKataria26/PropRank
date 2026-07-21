import sqlite3
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_training_pairs(feature_version_id, query_contexts, config):
    if not query_contexts:
        logger.warning("No query_contexts provided. Defaulting to all contexts. "
                       "This could silently mix unrelated ranking domains into one training run.")
        query_contexts = []
        
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Base query for preference logs
    query = """
        SELECT 
            p.item_i_id, p.item_j_id, p.winner_id, p.query_context, p.position_i, p.position_j, p.user_hash,
            i.feature_vector_json AS features_i,
            j.feature_vector_json AS features_j
        FROM preference_log p
        LEFT JOIN items i ON p.item_i_id = i.item_id AND i.feature_version_id = ?
        LEFT JOIN items j ON p.item_j_id = j.item_id AND j.feature_version_id = ?
        WHERE p.source IN ('simulated', 'real')
    """
    
    params = [feature_version_id, feature_version_id]
    
    if query_contexts:
        placeholders = ",".join(["?"] * len(query_contexts))
        query += f" AND p.query_context IN ({placeholders})"
        params.extend(query_contexts)
        
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    skipped_count = 0
    valid_pairs = []
    expected_dim = None
    
    for row in rows:
        features_i_str = row['features_i']
        features_j_str = row['features_j']
        
        if features_i_str is None or features_j_str is None:
            skipped_count += 1
            logger.info(f"Skipped pair ({row['item_i_id']}, {row['item_j_id']}): missing feature vector for given feature_version_id")
            continue
            
        try:
            feat_i = np.array(json.loads(features_i_str), dtype=np.float32)
            feat_j = np.array(json.loads(features_j_str), dtype=np.float32)
        except Exception as e:
            skipped_count += 1
            logger.error(f"Failed to parse JSON features: {e}")
            continue
            
        if expected_dim is None:
            expected_dim = feat_i.shape[0]
            
        if feat_i.shape[0] != expected_dim or feat_j.shape[0] != expected_dim:
            raise ValueError(f"Feature dimension mismatch! Expected {expected_dim}, got {feat_i.shape[0]} and {feat_j.shape[0]}")
            
        label = 1.0 if row['winner_id'] == row['item_i_id'] else 0.0
            
        valid_pairs.append({
            'item_i_id': row['item_i_id'],
            'item_j_id': row['item_j_id'],
            'item_i_features': feat_i,
            'item_j_features': feat_j,
            'label': label,
            'query_context': row['query_context'],
            'position_i': row['position_i'],
            'position_j': row['position_j'],
            'user_hash': row['user_hash']
        })
        
    if skipped_count > 0:
        logger.info(f"Total skipped pairs due to missing/invalid items: {skipped_count}")
        
    # Enforce source-diversity floor
    max_ratio = getattr(config, 'max_user_contribution_ratio', None)
    if max_ratio is not None and max_ratio > 0:
        filtered_pairs = []
        from collections import defaultdict
        pairs_by_context = defaultdict(list)
        for p in valid_pairs:
            pairs_by_context[p['query_context']].append(p)
            
        for ctx, pairs in pairs_by_context.items():
            max_allowed = int(len(pairs) * max_ratio)
            # If max_allowed is 0 but max_ratio > 0, we should at least allow 1 if it's a very small dataset?
            # Actually, if 5% of 10 is 0, then max_allowed = 0, meaning we discard everything?
            # Let's say max_allowed = max(1, int(len(pairs) * max_ratio))
            max_allowed = max(1, max_allowed)
            
            user_counts = defaultdict(int)
            for p in pairs:
                user = p['user_hash']
                if user is None:
                    # null user hashes don't count towards a single specific user limit
                    filtered_pairs.append(p)
                elif user_counts[user] < max_allowed:
                    user_counts[user] += 1
                    filtered_pairs.append(p)
        
        excluded_count = len(valid_pairs) - len(filtered_pairs)
        if excluded_count > 0:
            logger.info(f"Source-diversity floor excluded {excluded_count} pairs exceeding max user contribution of {max_ratio*100}% per context.")
        valid_pairs = filtered_pairs

    if len(valid_pairs) == 0:
        raise ValueError("Zero valid pairs remain after joining/filtering. Cannot proceed to train on an empty dataset.")
        
    return valid_pairs
