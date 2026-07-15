import math

def click_probability(position):
    """
    Returns the probability of an item being 'seen' (and potentially clicked) 
    at a given 1-indexed position.
    
    Formula: P(seen|position) = 1 / sqrt(position)
    
    Args:
        position (int): The 1-indexed position of the item in the ranking.
        
    Returns:
        float: The examination probability.
        
    Raises:
        ValueError: If position <= 0.
    """
    if position <= 0:
        raise ValueError("Position must be >= 1. Negative or zero positions are undefined/meaningless.")
    
    # As position approaches infinity, 1 / sqrt(position) approaches 0
    # but never exactly equals 0 or becomes negative.
    prob = 1.0 / math.sqrt(position)
    return prob
