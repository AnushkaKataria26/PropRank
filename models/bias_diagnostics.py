import json
import os
import torch
import numpy as np
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

def measure_position_rank_correlation(model, held_out_pairs, config):
    gt_path = "simulator/artifacts/ground_truth_relevance.json"
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            "ground_truth_relevance.json is missing. This specific diagnostic is only valid "
            "on simulated data where true relevance is known. Do not run this on real data without ground truth."
        )
        
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
        
    items_data = {}
    for p in held_out_pairs:
        ctx = p['query_context']
        i_id = p['item_i_id']
        j_id = p['item_j_id']
        i_pos = p['position_i']
        j_pos = p['position_j']
        
        i_key = f"{i_id}|{ctx}"
        j_key = f"{j_id}|{ctx}"
        
        if i_key not in items_data and i_pos is not None:
            items_data[i_key] = {
                'features': p['item_i_features'],
                'position': i_pos,
                'gt_relevance': ground_truth.get(i_key, None)
            }
        if j_key not in items_data and j_pos is not None:
            items_data[j_key] = {
                'features': p['item_j_features'],
                'position': j_pos,
                'gt_relevance': ground_truth.get(j_key, None)
            }
            
    # Filter items that lack ground truth (just in case)
    valid_items = [v for v in items_data.values() if v['gt_relevance'] is not None]
    
    if not valid_items:
        raise ValueError("No held-out items matched the ground truth relevance file.")
        
    features = [torch.tensor(v['features'], dtype=torch.float32) for v in valid_items]
    positions = [v['position'] for v in valid_items]
    gt_relevances = [v['gt_relevance'] for v in valid_items]
    
    model.eval()
    with torch.no_grad():
        feat_tensor = torch.stack(features)
        scores = model(feat_tensor).squeeze(-1).numpy()
        
    # (a) position vs (b) model predicted score
    pos_score_corr, _ = spearmanr(positions, scores)
    
    # (c) ground-truth relevance vs (d) predicted score
    rel_score_corr, _ = spearmanr(gt_relevances, scores)
    
    logger.info(f"Position vs Score correlation: {pos_score_corr:.4f}")
    logger.info(f"Relevance vs Score correlation: {rel_score_corr:.4f}")
    
    return pos_score_corr, rel_score_corr
