import json
import os
import hashlib

def assign_ground_truth_relevance(item_ids, query_contexts, config):
    """
    Assigns a ground-truth relevance score (0.0 to 1.0) to each (item_id, query_context) pair.
    
    This function uses a deterministic hash-based pseudo-random method seeded by 
    config.random_seed so that the relevance score is reproducible across runs.
    
    The resulting scores are NOT written to preference_log or items. They are 
    ground truth used only for simulation and later evaluation.
    
    Args:
        item_ids (list): List of candidate item IDs.
        query_contexts (list): List of query contexts.
        config (Config): Configuration object with a random_seed.
        
    Returns:
        dict: A mapping from (item_id, query_context) to a float score in [0.0, 1.0].
    """
    if not item_ids:
        raise ValueError("item_ids list cannot be empty.")
    if not query_contexts:
        raise ValueError("query_contexts list cannot be empty.")
        
    relevance_map = {}
    
    for qc in query_contexts:
        for iid in item_ids:
            # Deterministic hash to map (item_id, query_context, random_seed) to [0.0, 1.0]
            hash_input = f"{iid}_{qc}_{config.random_seed}".encode("utf-8")
            hash_hex = hashlib.sha256(hash_input).hexdigest()
            
            # Convert first 8 hex characters (32 bits) to a float in [0.0, 1.0)
            hash_int = int(hash_hex[:8], 16)
            score = hash_int / 0xffffffff
            
            relevance_map[(iid, qc)] = score
            
    # Persist it for Phase 4 artifact
    artifact_dir = os.path.join("simulator", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    artifact_path = os.path.join(artifact_dir, "ground_truth_relevance.json")
    
    # Convert tuple keys to string for JSON serialization
    json_ready_map = {f"{iid}|{qc}": score for (iid, qc), score in relevance_map.items()}
    
    with open(artifact_path, "w") as f:
        # NOT available in a real deployment — used only to validate bias correction in Phase 4.
        json.dump(json_ready_map, f, indent=2)
        
    return relevance_map
