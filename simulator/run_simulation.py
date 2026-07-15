import sqlite3
import logging
import os
from simulator.relevance import assign_ground_truth_relevance
from simulator.candidate_lists import generate_candidate_ranking
from simulator.simulate_clicks import simulate_preference_pairs
from simulator.persist import persist_preference_pairs

def run_simulation(query_contexts, num_pairs_per_context, config):
    """
    Entrypoint for the position-bias-aware click simulator.
    
    Args:
        query_contexts (list): List of query contexts to simulate.
        num_pairs_per_context (int): Number of preference pairs to generate per query.
        config (Config): Application configuration.
    """
    if not query_contexts:
        print("No query contexts provided. Exiting.")
        return
        
    db_path = config.db_path
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Verify items exist
    cursor.execute("SELECT item_id FROM items")
    rows = cursor.fetchall()
    conn.close()
    
    item_ids = [row[0] for row in rows]
    
    if not item_ids:
        raise ValueError("The 'items' table is empty. Cannot run simulation without items.")
        
    # Task 1: Ground truth relevance assignment
    relevance_map = assign_ground_truth_relevance(item_ids, query_contexts, config)
    
    total_generated = 0
    total_successful = 0
    total_failed = 0
    processed_queries = 0
    
    for qc in query_contexts:
        # Task 3: Ranked candidate list generation
        ranking = generate_candidate_ranking(item_ids, qc, relevance_map, config)
        if not ranking:
            continue
            
        # Task 4: Click/preference simulation
        pairs = simulate_preference_pairs(ranking, relevance_map, qc, config, num_pairs_per_context)
        
        # Task 5: Persist pairs
        if pairs:
            total_generated += len(pairs)
            succ, fail = persist_preference_pairs(pairs, config)
            total_successful += succ
            total_failed += fail
            
        processed_queries += 1
        
    print("--- Simulation Summary ---")
    print(f"Query contexts processed: {processed_queries}")
    print(f"Total pairs generated:    {total_generated}")
    print(f"Total pairs saved:        {total_successful}")
    print(f"Total pairs skipped:      {total_failed}")
    print(f"Artifact saved to:        simulator/artifacts/ground_truth_relevance.json")
    print("--------------------------")
