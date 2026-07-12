import os
import sqlite3
import pytest
from unittest.mock import patch
import json

from config.loader import Config
from db.init_db import init_db

@pytest.fixture
def temp_config(tmp_path):
    db_path = tmp_path / "test_propRank.sqlite3"
    config_dict = {
        "tfidf_max_features": 500,
        "train_batch_size": 64,
        "train_epochs": 100,
        "retrain_pair_threshold": 500,
        "confidence_pair_threshold": 100,
        "ndcg_retrain_floor": 0.60,
        "held_out_split_ratio": 0.20,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": str(db_path),
        "random_seed": 42
    }
    
    config = Config(config_dict)
    
    # Mock get_config to return this temp config
    with patch("db.init_db.get_config", return_value=config):
        yield config

def test_init_db_creates_tables(temp_config):
    db_path = temp_config.db_path
    
    # Ensure db does not exist initially
    assert not os.path.exists(db_path)
    
    # Run init_db
    init_db()
    
    # Verify DB file created
    assert os.path.exists(db_path)
    
    # Connect and check tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}
    
    expected_tables = {"items", "preference_log", "model_versions", "feature_versions", "sqlite_sequence"}
    assert expected_tables.issubset(tables)
    
    # Close connection
    conn.close()

def test_init_db_idempotent(temp_config, capsys):
    # Run twice
    init_db()
    
    # Capture output before second run
    capsys.readouterr()
    
    init_db()
    
    # Should print a warning
    captured = capsys.readouterr()
    assert "Warning: Database at" in captured.out
    assert "already exists" in captured.out

def test_preference_log_check_constraints(temp_config):
    init_db()
    conn = sqlite3.connect(temp_config.db_path)
    cursor = conn.cursor()
    
    # Valid insert
    try:
        cursor.execute('''
            INSERT INTO preference_log 
            (item_i_id, item_j_id, winner_id, query_context, source) 
            VALUES (?, ?, ?, ?, ?)
        ''', ("item_1", "item_2", "item_1", "query_1", "simulated"))
        conn.commit()
    except sqlite3.Error as e:
        pytest.fail(f"Valid insert failed: {e}")
        
    # Invalid insert: winner_id not item_i or item_j
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute('''
            INSERT INTO preference_log 
            (item_i_id, item_j_id, winner_id, query_context, source) 
            VALUES (?, ?, ?, ?, ?)
        ''', ("item_1", "item_2", "item_3", "query_1", "simulated"))
        
    # Invalid insert: self-pair
    with pytest.raises(sqlite3.IntegrityError):
        cursor.execute('''
            INSERT INTO preference_log 
            (item_i_id, item_j_id, winner_id, query_context, source) 
            VALUES (?, ?, ?, ?, ?)
        ''', ("item_1", "item_1", "item_1", "query_1", "simulated"))
        
    conn.close()
