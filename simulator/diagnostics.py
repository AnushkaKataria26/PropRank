import sqlite3
import scipy.stats
import os

def measure_position_bias(config):
    """
    Reads simulated pairs from the database and computes the correlation 
    between an item's position and whether it won the pairwise comparison.
    
    A strong negative correlation is expected (lower position number -> higher win rate).
    We'll print the diagnostic metrics.
    
    Args:
        config (Config): Configuration object containing db_path.
        
    Returns:
        float: Spearman correlation coefficient.
    """
    db_path = config.db_path
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return 0.0
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # We want to know for each item in the comparison: its position, and whether it won (1 or 0)
    query = """
    SELECT 
        position_i, (CASE WHEN winner_id = item_i_id THEN 1 ELSE 0 END) as i_won,
        position_j, (CASE WHEN winner_id = item_j_id THEN 1 ELSE 0 END) as j_won
    FROM preference_log
    WHERE source = 'simulated'
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No simulated pairs found in preference_log.")
        return 0.0
        
    positions = []
    wins = []
    
    for row in rows:
        # Item I
        positions.append(row[0])
        wins.append(row[1])
        # Item J
        positions.append(row[2])
        wins.append(row[3])
        
    # We use Spearman rank correlation
    correlation, p_value = scipy.stats.spearmanr(positions, wins)
    
    print("--- Diagnostic: Position Bias ---")
    print(f"Total simulated items observed: {len(positions)}")
    print(f"Spearman correlation (Position vs Win): {correlation:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    if correlation < -0.1:
        print("Result: STRONG POSITION BIAS DETECTED. (Simulator is working as intended)")
    elif correlation < 0:
        print("Result: WEAK POSITION BIAS DETECTED. (Consider increasing simulation pairs)")
    else:
        print("Result: NO POSITION BIAS DETECTED. (Warning: Something might be wrong with simulator)")
    print("---------------------------------")
    
    return correlation
