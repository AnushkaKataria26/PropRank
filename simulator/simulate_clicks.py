import math
import hashlib
import numpy as np
from simulator.click_model import click_probability

def simulate_preference_pairs(candidate_ranking, ground_truth_relevance, query_context, config, num_pairs_per_context):
    """
    Samples pairs of items from the candidate ranking and simulates a pairwise preference.
    
    The probabilistic model:
    (a) Position-based click_probability determines whether each item is "seen".
    (b) If both are seen, a Bradley-Terry-style comparison on ground-truth relevance 
        determines the winner: P(i wins | seen both) = exp(rel_i) / (exp(rel_i) + exp(rel_j)).
    (c) If only one item is "seen", it wins regardless of true relevance.
    (d) If neither is seen, the pair is resampled until we get a valid observation.
    
    If items have identical relevance and both are seen, a fair coin flip decides the winner.
    
    Args:
        candidate_ranking (list): Ordered list of item_ids (position 1 is index 0).
        ground_truth_relevance (dict): Mapping from (item_id, query_context) to float score.
        query_context (str): The query context string.
        config (Config): Configuration object containing random_seed.
        num_pairs_per_context (int): Number of pairs to generate.
        
    Returns:
        list: A list of dicts representing the generated preference pairs.
    """
    pairs = []
    n_candidates = len(candidate_ranking)
    if n_candidates < 2:
        return pairs
        
    # Generate a reproducible random seed for sampling pairs
    hash_input = f"{query_context}_{config.random_seed}_pairs".encode("utf-8")
    hash_int = int(hashlib.sha256(hash_input).hexdigest()[:8], 16)
    rng = np.random.RandomState(hash_int)
    
    while len(pairs) < num_pairs_per_context:
        # Sample two distinct positions
        idx_i, idx_j = rng.choice(n_candidates, size=2, replace=False)
        
        item_i_id = candidate_ranking[idx_i]
        item_j_id = candidate_ranking[idx_j]
        
        pos_i = int(idx_i) + 1
        pos_j = int(idx_j) + 1
        
        rel_i = ground_truth_relevance[(item_i_id, query_context)]
        rel_j = ground_truth_relevance[(item_j_id, query_context)]
        
        prob_seen_i = click_probability(pos_i)
        prob_seen_j = click_probability(pos_j)
        
        seen_i = rng.rand() < prob_seen_i
        seen_j = rng.rand() < prob_seen_j
        
        winner_id = None
        
        if seen_i and seen_j:
            if math.isclose(rel_i, rel_j):
                # Tie breaker
                winner_id = item_i_id if rng.rand() < 0.5 else item_j_id
            else:
                # Bradley-Terry on ground-truth relevance
                # Formula: exp(rel_i) / (exp(rel_i) + exp(rel_j))
                p_i_wins = math.exp(rel_i) / (math.exp(rel_i) + math.exp(rel_j))
                winner_id = item_i_id if rng.rand() < p_i_wins else item_j_id
        elif seen_i and not seen_j:
            winner_id = item_i_id
        elif seen_j and not seen_i:
            winner_id = item_j_id
        else:
            # Neither seen, resample
            continue
            
        pairs.append({
            "item_i_id": item_i_id,
            "item_j_id": item_j_id,
            "winner_id": winner_id,
            "query_context": query_context,
            "position_i": pos_i,
            "position_j": pos_j
        })
        
    return pairs
