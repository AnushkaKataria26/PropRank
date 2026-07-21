import sys
import sqlite3
from config.loader import get_config
from simulator.run_simulation import run_simulation
from models.run_training_corrected import run_training_pipeline_corrected
from inference.run_inference import run_inference
from demo.setup_demo_data import setup_demo_data

def run_scenario_2():
    print("\n--- Scenario 2: Warm Context / Confident Ranking ---")
    config = get_config()
    
    # Check if DB is initialized. setup_demo_data does this.
    feature_version_id, contexts = setup_demo_data()
    warm_context = contexts['warm']
    
    threshold = config.confidence_pair_threshold
    print(f"Targeting > {threshold} pairs for context: {warm_context}")
    
    # 1. Generate and persist preference pairs
    # Wait, we should generate at least threshold + some padding
    run_simulation([warm_context], threshold + 20, config)
    
    # Verify count via a direct query
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM preference_log WHERE query_context = ?", (warm_context,))
    count = cursor.fetchone()[0]
    
    # Also check if there's an active model
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE is_active = 1")
    active_models = cursor.fetchone()[0]
    conn.close()
    
    print(f"Persisted pairs for {warm_context}: {count}")
    assert count > threshold, f"Failed to generate enough pairs. Got {count}, need {threshold}."
    
    # 2. Run Phase 4 training if no suitably trained active model already covers it
    if active_models == 0:
        print("No active model found. Training a new bias-corrected model...")
        new_model_version_id = run_training_pipeline_corrected(feature_version_id, [warm_context], config)
        print(f"Trained new model_version_id: {new_model_version_id}")
    else:
        print("Active model already exists. Proceeding to inference.")
        
    try:
        results = run_inference(warm_context, 10, config)
        
        if not results:
            print("No results returned.")
            return False
            
        confidence_flag = results[0].get('confidence_flag')
        fallback_used = results[0].get('fallback_used')
        
        print(f"Results: {[r['item_id'] for r in results[:5]]}")
        print(f"Confidence Flag: {confidence_flag}")
        print(f"Fallback Used: {fallback_used}")
        
        assert confidence_flag == 'CONFIDENT', f"Expected CONFIDENT, got {confidence_flag}"
        assert fallback_used is False, f"Expected fallback_used=False, got {fallback_used}"
        
        print("Scenario 2 PASSED.\n")
        return True
    except Exception as e:
        print(f"Scenario 2 FAILED: {e}\n")
        return False

if __name__ == "__main__":
    success = run_scenario_2()
    if not success:
        sys.exit(1)
