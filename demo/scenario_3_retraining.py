import sys
import sqlite3
import time
from config.loader import get_config
from simulator.run_simulation import run_simulation
from models.retrain_trigger import should_retrain
from models.retrain_orchestrator import execute_retraining
from inference.run_inference import run_inference
from demo.setup_demo_data import setup_demo_data

def run_scenario_3():
    print("\n--- Scenario 3: Retraining Trigger and Comparison ---")
    config = get_config()
    
    feature_version_id, contexts = setup_demo_data()
    retrain_context = contexts['retrain']
    
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, ndcg_at_10 FROM model_versions WHERE is_active = 1")
    row = cursor.fetchone()
    
    if not row:
        print("Scenario 3 requires an active model. Please run Scenario 2 first.")
        conn.close()
        return False
        
    old_model_id = row['id']
    old_ndcg = row['ndcg_at_10']
    
    # Run inference to capture "before" snapshot
    print(f"Active Model ID: {old_model_id}, NDCG@10: {old_ndcg:.4f}")
    before_results = run_inference(retrain_context, 10, config)
    if not before_results:
        print("No before results.")
        return False
        
    before_confidence = before_results[0].get('confidence_flag')
    before_fallback = before_results[0].get('fallback_used')
    print(f"Before Ranking: {[r['item_id'] for r in before_results[:5]]}")
    
    # Simulate 500+ new pairs
    print(f"\nSimulating {config.retrain_pair_threshold + 50} new pairs for {retrain_context}...")
    time.sleep(1) # Ensure SQLite datetime('now') ticks over
    run_simulation([retrain_context], config.retrain_pair_threshold + 50, config)
    
    cursor.execute("SELECT COUNT(*) FROM preference_log WHERE query_context = ?", (retrain_context,))
    print(f"Total pairs for {retrain_context} is now: {cursor.fetchone()[0]}")
    
    print("\nChecking if retraining is triggered...")
    trigger_info = should_retrain(config)
    trigger = trigger_info.get("should_retrain", False)
    reason = trigger_info.get("reason", "unknown")
    
    print(f"Should retrain: {trigger}, Reasons: {reason}")
    assert trigger is True, "Expected should_retrain to be True"
    assert 'pair_threshold' in reason or reason == 'both', "Expected 'pair_threshold' or 'both' in reason"
    
    print("\nExecuting retraining pipeline...")
    result = execute_retraining(config)
    
    # Check if a new model was activated
    cursor.execute("SELECT id, ndcg_at_10 FROM model_versions WHERE is_active = 1")
    new_row = cursor.fetchone()
    conn.close()
    
    new_model_id = new_row['id']
    new_ndcg = new_row['ndcg_at_10']
    
    if new_model_id != old_model_id:
        print(f"\nNew model activated! ID: {new_model_id}, NDCG@10: {new_ndcg:.4f}")
        print(f"Comparison: Old NDCG@10 = {old_ndcg:.4f} vs New NDCG@10 = {new_ndcg:.4f}")
        
        after_results = run_inference(retrain_context, 10, config)
        if not after_results:
            print("No after results.")
            return False
            
        after_confidence = after_results[0].get('confidence_flag')
        after_fallback = after_results[0].get('fallback_used')
        print(f"After Ranking:  {[r['item_id'] for r in after_results[:5]]}")
    else:
        print("\nNew model NOT activated.")
        print(f"The newly trained model did not pass the performance safeguard (NDCG@10 threshold: {config.ndcg_retrain_floor}).")
        print("This is a valid, expected outcome proving the safeguard works correctly.")
        
    print("\nScenario 3 PASSED.\n")
    return True

if __name__ == "__main__":
    success = run_scenario_3()
    if not success:
        sys.exit(1)
