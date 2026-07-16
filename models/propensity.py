import numpy as np
import logging
from simulator.click_model import click_probability

logger = logging.getLogger(__name__)

def compute_propensity_weights(pairs, config):
    """
    Computes inverse propensity weights for position bias correction.
    
    The weight is calculated as 1 / click_probability(position_of_winner).
    The winning item's position is used because IPW corrects for the 
    probability of the preferred item actually being examined/seen by the user.
    """
    weights = []
    fallback_count = 0
    clip_count = 0
    max_weight_before_clip = 0.0
    
    for pair in pairs:
        # Determine the position of the winning item
        if pair['label'] == 1.0:
            relevant_pos = pair['position_i']
        else:
            relevant_pos = pair['position_j']
            
        # Fallback for missing position data
        if relevant_pos is None:
            weights.append(1.0)
            fallback_count += 1
            continue
            
        # Calculate propensity weight
        prob = click_probability(relevant_pos)
        weight = 1.0 / prob
        
        # Track pre-clip maximum for diagnostics
        if weight > max_weight_before_clip:
            max_weight_before_clip = weight
            
        # Clip extreme weights to prevent instability
        if weight > config.propensity_weight_clip:
            weight = float(config.propensity_weight_clip)
            clip_count += 1
            
        weights.append(weight)
        
    weights = np.array(weights, dtype=np.float32)
    
    if fallback_count > 0:
        logger.info(f"Fallback to weight 1.0 for {fallback_count} pairs due to missing position.")
        
    if clip_count > 0:
        logger.info(f"Clipped {clip_count} weights at {config.propensity_weight_clip}. "
                    f"Max pre-clip weight was {max_weight_before_clip:.4f}.")
                    
    # Normalize weights so they average to 1.0
    # This ensures the effective learning rate scale remains consistent with the unweighted baseline
    if config.propensity_normalize and len(weights) > 0:
        mean_weight = np.mean(weights)
        if mean_weight > 0:
            weights = weights / mean_weight
            logger.info(f"Normalized weights. Original mean was {mean_weight:.4f}.")
            
    stats = {
        'fallback_count': fallback_count,
        'clip_count': clip_count,
        'max_weight_before_clip': max_weight_before_clip
    }
            
    return weights, stats
