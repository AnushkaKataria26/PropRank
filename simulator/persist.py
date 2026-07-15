import sqlite3
import logging

def persist_preference_pairs(pairs, config, synth_timestamp=None):
    """
    Persists simulated preference pairs to the preference_log table.
    
    Args:
        pairs (list): List of dicts with keys (item_i_id, item_j_id, winner_id, 
                      query_context, position_i, position_j).
        config (Config): Configuration object containing db_path.
        synth_timestamp (str, optional): An optional synthetic timestamp to override 
                                         the default (current time).
                                         
    Returns:
        tuple: (int, int) representing (successful_inserts, skipped_or_failed_inserts).
    """
    if not pairs:
        return 0, 0
        
    db_path = config.db_path
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    successful = 0
    failed = 0
    
    query = """
    INSERT INTO preference_log (
        item_i_id, item_j_id, winner_id, query_context, 
        position_i, position_j, source, user_hash
        {timestamp_col}
    ) VALUES (
        ?, ?, ?, ?, ?, ?, 'simulated', NULL
        {timestamp_val}
    )
    """
    
    if synth_timestamp:
        query = query.format(timestamp_col=", timestamp", timestamp_val=", ?")
    else:
        # Use DB default (datetime('now'))
        query = query.format(timestamp_col="", timestamp_val="")
    
    # We use a single transaction. Instead of executemany which would abort the batch 
    # on a single IntegrityError, we iterate and catch row-level exceptions.
    cursor.execute("BEGIN TRANSACTION")
    
    for pair in pairs:
        try:
            params = [
                pair["item_i_id"],
                pair["item_j_id"],
                pair["winner_id"],
                pair["query_context"],
                pair["position_i"],
                pair["position_j"]
            ]
            if synth_timestamp:
                params.append(synth_timestamp)
                
            cursor.execute(query, params)
            successful += 1
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error for pair {pair}: {e}")
            failed += 1
        except Exception as e:
            logging.error(f"Unexpected error for pair {pair}: {e}")
            failed += 1
            
    conn.commit()
    conn.close()
    
    return successful, failed
