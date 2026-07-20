import json
import numpy as np
import torch

def score_with_model(bundle, candidate_items, config):
    """
    Scores candidates using the provided model bundle.
    
    Returns: list of (item_id, score) sorted descending by score.
    """
    expected_dim = bundle.model.mlp[0].in_features
    
    vectors = []
    item_ids = []
    
    for item in candidate_items:
        item_id = item['item_id']
        feat_json = item.get('feature_vector_json')
        
        if feat_json is None:
            raise ValueError(f"Item {item_id} is missing feature_vector_json.")
            
        try:
            feat_array = np.array(json.loads(feat_json), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse feature_vector_json for item {item_id}: {e}")
            
        if feat_array.shape[0] != expected_dim:
            raise ValueError(f"Dimension mismatch for item {item_id}: Expected {expected_dim}, got {feat_array.shape[0]}.")
            
        if np.isnan(feat_array).any():
            raise ValueError(f"NaN detected in feature vector for item {item_id}.")
            
        vectors.append(feat_array)
        item_ids.append(item_id)
        
    if not vectors:
        return []
        
    # Batch scoring
    batch_tensor = torch.tensor(np.stack(vectors), dtype=torch.float32)
    
    with torch.no_grad():
        scores = bundle.model(batch_tensor).squeeze(-1).numpy()
        
    scored_items = [
        (item_id, float(score)) 
        for item_id, score in zip(item_ids, scores)
    ]
    
    # Sort descending by score, tie-break by item_id
    scored_items.sort(key=lambda x: (-x[1], str(x[0])))
    
    return scored_items
