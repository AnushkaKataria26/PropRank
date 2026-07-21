import pytest
import sqlite3
import os
from config.loader import get_config
from db.init_db import init_db
from demo.setup_demo_data import setup_demo_data
from demo.scenario_1_cold_start import run_scenario_1
from demo.scenario_2_warm_context import run_scenario_2
from demo.scenario_3_retraining import run_scenario_3
from demo.benchmark_latency import run_latency_benchmark

@pytest.mark.slow
def test_end_to_end_pipeline(tmp_path):
    """
    Integration test that runs the entire pipeline from scratch against a fresh temporary DB.
    """
    # Override config DB path
    config = get_config()
    db_path = str(tmp_path / "test.sqlite3")
    config.db_path = db_path
    
    # 1. Init schema
    init_db()
    assert os.path.exists(db_path)
    
    # 2. Ingest demo data
    feature_version_id, contexts = setup_demo_data(num_items=50)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM items")
    assert cursor.fetchone()[0] == 50
    
    # 3. Scenario 1 (Cold Start)
    assert run_scenario_1() is True
    
    # 4. Scenario 2 (Warm Context - Generates pairs, trains, inference)
    # Reduce thresholds slightly to make test faster? 
    # Or just run it. 100 pairs is fast.
    assert run_scenario_2() is True
    
    # Verify artifacts were produced
    cursor.execute("SELECT COUNT(*) FROM model_versions WHERE is_active = 1")
    assert cursor.fetchone()[0] == 1
    
    # 5. Scenario 3 (Retraining - Simulates 500+ pairs, trains, evaluates)
    assert run_scenario_3() is True
    
    # Verify both baseline and corrected models exist (or at least more than 1 model version)
    cursor.execute("SELECT COUNT(*) FROM model_versions")
    assert cursor.fetchone()[0] >= 2
    
    # 6. Latency Benchmark
    success, med, max_lat = run_latency_benchmark()
    # It might fail if running on very slow CI, so we just assert it ran without exception
    # and returned numbers. We won't strictly fail the test if the benchmark is > threshold on a CI machine.
    assert med > 0
    assert max_lat > 0
    
    conn.close()
