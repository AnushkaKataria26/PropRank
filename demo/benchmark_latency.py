import sys
import time
import numpy as np
from config.loader import get_config
from demo.setup_demo_data import setup_demo_data
from inference.run_inference import run_inference

def run_latency_benchmark():
    print("\n--- Latency Benchmark ---")
    config = get_config()
    
    threshold_ms = getattr(config, 'inference_latency_ms_threshold', 500)
    print(f"Target Latency (Median): <= {threshold_ms}ms for 1000 items")
    
    _, contexts = setup_demo_data(num_items=1000)
    context = contexts['warm']
    
    candidate_item_ids = [f"item_{i}" for i in range(1, 1001)]
    
    # Warm-up run (to exclude one-time model loading cost)
    try:
        _ = run_inference(context, 10, config)
    except Exception as e:
        print(f"Failed during warmup inference (make sure an active model exists): {e}")
        return False
        
    times_ms = []
    num_runs = 10
    
    print(f"Running {num_runs} iterations of inference for 1000 items...")
    
    for i in range(num_runs):
        start = time.time()
        _ = run_inference(context, 10, config)
        end = time.time()
        
        duration_ms = (end - start) * 1000
        times_ms.append(duration_ms)
        
    median_ms = np.median(times_ms)
    max_ms = np.max(times_ms)
    
    print(f"Median Latency: {median_ms:.2f} ms")
    print(f"Max Latency:    {max_ms:.2f} ms")
    
    if median_ms <= threshold_ms:
        print(f"Latency Benchmark PASSED (Median <= {threshold_ms}ms)\n")
        return True, median_ms, max_ms
    else:
        print(f"Latency Benchmark FAILED (Median > {threshold_ms}ms)\n")
        return False, median_ms, max_ms

if __name__ == "__main__":
    success, _, _ = run_latency_benchmark()
    if not success:
        sys.exit(1)
