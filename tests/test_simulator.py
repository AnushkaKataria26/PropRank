import pytest
import math
import sqlite3
import os
import json
from simulator.click_model import click_probability
from simulator.relevance import assign_ground_truth_relevance
from simulator.candidate_lists import generate_candidate_ranking
from simulator.simulate_clicks import simulate_preference_pairs
from simulator.persist import persist_preference_pairs
from simulator.run_simulation import run_simulation
from simulator.diagnostics import measure_position_bias
from config.loader import Config

@pytest.fixture
def dummy_config():
    return Config({
        "tfidf_max_features": 500,
        "train_batch_size": 64,
        "train_epochs": 100,
        "retrain_pair_threshold": 500,
        "confidence_pair_threshold": 100,
        "ndcg_retrain_floor": 0.60,
        "held_out_split_ratio": 0.20,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": "db/test_propRank.sqlite3",
        "random_seed": 42
    })

def test_click_probability_boundary():
    with pytest.raises(ValueError):
        click_probability(0)
    with pytest.raises(ValueError):
        click_probability(-1)
        
def test_click_probability_monotonic():
    p1 = click_probability(1)
    p2 = click_probability(2)
    p100 = click_probability(100)
    p1000 = click_probability(1000)
    
    assert p1 == 1.0
    assert p1 > p2 > p100 > p1000
    assert p1000 > 0.0

def test_relevance_reproducibility(dummy_config):
    item_ids = ["item1", "item2", "item3"]
    query_contexts = ["q1", "q2"]
    
    rel1 = assign_ground_truth_relevance(item_ids, query_contexts, dummy_config)
    rel2 = assign_ground_truth_relevance(item_ids, query_contexts, dummy_config)
    
    assert rel1 == rel2
    
    # Check bounds
    for v in rel1.values():
        assert 0.0 <= v <= 1.0
        
def test_candidate_ranking_edge_case(dummy_config, caplog):
    item_ids = ["item1"] # fewer than 2 items
    rel_map = {("item1", "q1"): 0.5}
    ranking = generate_candidate_ranking(item_ids, "q1", rel_map, dummy_config)
    
    assert ranking == []
    assert "fewer than 2 items" in caplog.text

def test_simulate_preference_pairs_no_self_comparison(dummy_config):
    # Setup candidate ranking with 5 items
    candidate_ranking = ["item1", "item2", "item3", "item4", "item5"]
    rel_map = {(i, "q1"): 0.5 for i in candidate_ranking}
    
    # Generate many pairs to ensure we never get item_i_id == item_j_id
    pairs = simulate_preference_pairs(candidate_ranking, rel_map, "q1", dummy_config, 100)
    
    for pair in pairs:
        assert pair["item_i_id"] != pair["item_j_id"]
        assert pair["winner_id"] in (pair["item_i_id"], pair["item_j_id"])

def test_simulate_preference_pairs_tie_handling(dummy_config):
    candidate_ranking = ["item1", "item2"]
    # Identical relevance
    rel_map = {("item1", "q1"): 0.5, ("item2", "q1"): 0.5}
    
    # Should not crash
    pairs = simulate_preference_pairs(candidate_ranking, rel_map, "q1", dummy_config, 10)
    assert len(pairs) == 10
    
def test_persist_graceful_failure(dummy_config, tmp_path):
    db_path = str(tmp_path / "test.db")
    dummy_config.db_path = db_path
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE preference_log (
        id INTEGER PRIMARY KEY, item_i_id TEXT, item_j_id TEXT, winner_id TEXT,
        query_context TEXT, position_i INTEGER, position_j INTEGER,
        source TEXT, user_hash TEXT, timestamp TEXT,
        CHECK (winner_id = item_i_id OR winner_id = item_j_id),
        CHECK (item_i_id != item_j_id)
    )
    """)
    conn.close()
    
    valid_pair = {
        "item_i_id": "i1", "item_j_id": "i2", "winner_id": "i1",
        "query_context": "q1", "position_i": 1, "position_j": 2
    }
    
    invalid_pair_1 = {
        "item_i_id": "i1", "item_j_id": "i1", "winner_id": "i1", # fails i != j
        "query_context": "q1", "position_i": 1, "position_j": 2
    }
    
    invalid_pair_2 = {
        "item_i_id": "i3", "item_j_id": "i4", "winner_id": "i99", # fails winner matching
        "query_context": "q1", "position_i": 1, "position_j": 2
    }
    
    pairs = [valid_pair, invalid_pair_1, valid_pair, invalid_pair_2, valid_pair]
    succ, fail = persist_preference_pairs(pairs, dummy_config)
    
    assert succ == 3
    assert fail == 2

def test_end_to_end_bias_diagnostic(dummy_config, tmp_path):
    db_path = str(tmp_path / "test.db")
    dummy_config.db_path = db_path
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS preference_log (
        id INTEGER PRIMARY KEY, item_i_id TEXT, item_j_id TEXT, winner_id TEXT,
        query_context TEXT, position_i INTEGER, position_j INTEGER,
        source TEXT, user_hash TEXT, timestamp TEXT DEFAULT (datetime('now')),
        CHECK (winner_id = item_i_id OR winner_id = item_j_id),
        CHECK (item_i_id != item_j_id)
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY, item_id TEXT UNIQUE
    )
    """)
    # Insert 10 synthetic items
    for i in range(10):
        conn.execute("INSERT INTO items (item_id) VALUES (?)", (f"item_{i}",))
    conn.commit()
    conn.close()
    
    # Run simulation
    run_simulation(["test_query_1", "test_query_2"], 500, dummy_config)
    
    # Measure bias
    correlation = measure_position_bias(dummy_config)
    
    # Should be negative because lower position number (1 is best) correlates with higher win rate (1 for win, 0 for loss)
    # Actually wait. Positions are 1 to 10. Win is 1 or 0.
    # Lower position (e.g. 1) -> more wins (1). Higher position (e.g. 10) -> fewer wins (0).
    # This means as position increases, win rate decreases. 
    # Therefore, correlation between position and win should be NEGATIVE.
    assert correlation < -0.05
