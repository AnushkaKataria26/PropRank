import logging
import hashlib
import numpy as np

def generate_candidate_ranking(item_ids, query_context, ground_truth_relevance, config):
    """
    Produces an initial "shown" ranking of candidate items for a query context.
    
    To create realistic position bias, this initial ranking is NOT sorted purely 
    by ground-truth relevance. If it were, position and relevance would be perfectly 
    correlated and impossible to disentangle. Instead, we use a noisy proxy ranking:
    ground truth relevance + injected Gaussian noise. This ensures position and true 
    relevance are correlated but not identical, making bias correction verifiable later.
    
    Args:
        item_ids (list): List of candidate item IDs.
        query_context (str): The query context string.
        ground_truth_relevance (dict): Mapping from (item_id, query_context) to float score.
        config (Config): Configuration object containing random_seed.
        
    Returns:
        list: Ordered list of item_ids representing the ranking (index 0 is position 1).
              Returns empty list if fewer than 2 items are available.
    """
    if len(item_ids) < 2:
        logging.warning(f"Query context '{query_context}' has fewer than 2 items. Skipping.")
        return []
        
    # Generate a reproducible random seed for this specific query context
    hash_input = f"{query_context}_{config.random_seed}".encode("utf-8")
    hash_int = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    
    rng = np.random.RandomState(hash_int)
    
    scored_items = []
    for item_id in item_ids:
        rel = ground_truth_relevance[(item_id, query_context)]
        # Add N(0, 0.5) noise to create a noisy proxy for ranking
        noise = rng.normal(0, 0.5)
        proxy_score = rel + noise
        scored_items.append((proxy_score, item_id))
        
    # Sort descending by proxy score
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # Return just the ordered item IDs
    return [item_id for _, item_id in scored_items]
