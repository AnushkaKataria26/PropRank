import sys
from config.loader import get_config
from demo.setup_demo_data import setup_demo_data
from inference.run_inference import run_inference
from models.run_training_corrected import run_training_pipeline_corrected
from simulator.run_simulation import run_simulation

def run_scenario_1():
    print("\n--- Scenario 1: Cold Start / Fallback ---")
    config = get_config()
    feature_version_id, contexts = setup_demo_data()
    cold_context = contexts['cold']
    warm_context = contexts['warm']
    
    
    # Ensure there is an active model by training on a warm context
    print("Simulating data and training initial model on warm context to enable inference engine...")
    run_simulation([warm_context], 100, config)
    run_training_pipeline_corrected(feature_version_id, [warm_context], config)
    
    # Run inference for a user query on a COLD context
    candidate_item_ids = [f"item_{i}" for i in range(1, 21)]
    
    try:
        results = run_inference(cold_context, 10, config)
        
        if not results:
            print("No results returned.")
            return False
            
        confidence_flag = results[0].get('confidence_flag')
        fallback_used = results[0].get('fallback_used')
        
        print(f"Results: {[r['item_id'] for r in results[:5]]}")
        print(f"Confidence Flag: {confidence_flag}")
        print(f"Fallback Used: {fallback_used}")
        
        assert confidence_flag == 'FALLBACK', f"Expected FALLBACK, got {confidence_flag}"
        assert fallback_used is True, f"Expected fallback_used=True, got {fallback_used}"
        
        print("Scenario 1 PASSED.\n")
        return True
    except Exception as e:
        print(f"Scenario 1 FAILED: {e}\n")
        return False

if __name__ == "__main__":
    success = run_scenario_1()
    if not success:
        sys.exit(1)
