import sqlite3

def check_context_confidence(query_context, config):
    """
    Checks whether we have enough historical data for the given query_context to 
    be confident in the model's predictions for it.
    
    This is a data-coverage confidence criterion, not a probability/score-based one.
    This is deliberate, since softmax/sigmoid outputs on pairwise scores are not 
    well-calibrated as absolute confidence estimates.
    """
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) 
        FROM preference_log 
        WHERE query_context = ?
    """, (query_context,))
    
    count = cursor.fetchone()[0]
    conn.close()
    
    if count >= config.confidence_pair_threshold:
        return 'CONFIDENT'
    return 'FALLBACK'
