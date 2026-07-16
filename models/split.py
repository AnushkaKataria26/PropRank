import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def split_pairs(loaded_pairs, config):
    # Group by query_context
    context_to_pairs = defaultdict(list)
    for pair in loaded_pairs:
        context_to_pairs[pair['query_context']].append(pair)
        
    train_pairs = []
    held_out_pairs = []
    
    rng = np.random.RandomState(config.random_seed)
    split_ratio = config.held_out_split_ratio
    
    for context, pairs in context_to_pairs.items():
        if len(pairs) < 5:
            logger.warning(f"Query context '{context}' has too few pairs ({len(pairs)}). "
                           "Putting all in train. No held-out evaluation coverage for this context.")
            train_pairs.extend(pairs)
        else:
            # shuffle pairs for this context
            pairs_arr = np.array(pairs, dtype=object)
            rng.shuffle(pairs_arr)
            
            num_held_out = int(len(pairs) * split_ratio)
            if num_held_out == 0 and split_ratio > 0:
                num_held_out = 1
                
            train_pairs.extend(pairs_arr[num_held_out:].tolist())
            held_out_pairs.extend(pairs_arr[:num_held_out].tolist())
            
    return train_pairs, held_out_pairs
