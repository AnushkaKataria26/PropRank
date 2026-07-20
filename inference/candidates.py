import sqlite3

def get_candidates_for_context(query_context, feature_version_id, config):
    """
    Retrieves all available items matching the given feature_version_id from the database.
    Note: candidates are not currently pre-filtered by query_context at the SQL level.
    Relevance is determined entirely by the model/BM25 scoring step. This is a scoping 
    decision for the MVP.
    
    Returns: A list of dicts representing the candidates.
    """
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT item_id, text_description, feature_vector_json 
        FROM items 
        WHERE feature_version_id = ?
    """, (feature_version_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError(f"No items found for feature_version_id {feature_version_id}.")
        
    candidates = []
    for row in rows:
        candidates.append({
            'item_id': row['item_id'],
            'text_description': row['text_description'],
            'feature_vector_json': row['feature_vector_json']
        })
        
    return candidates
