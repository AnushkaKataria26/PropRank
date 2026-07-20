import pytest
import sqlite3
import json
import os
from unittest.mock import patch, MagicMock

from config.loader import Config
from models.retrain_trigger import should_retrain
from models.retrain_orchestrator import execute_retraining

@pytest.fixture
def mock_config(tmp_path):
    db_path = tmp_path / "test_propRank.sqlite3"
    config_dict = {
        "tfidf_max_features": 100,
        "train_batch_size": 32,
        "train_epochs": 1,
        "retrain_pair_threshold": 10,
        "confidence_pair_threshold": 5,
        "ndcg_retrain_floor": 0.60,
        "held_out_split_ratio": 0.2,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": str(db_path),
        "random_seed": 42
    }
    return Config(config_dict)

@pytest.fixture
def test_db(mock_config):
    conn = sqlite3.connect(mock_config.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE feature_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tfidf_artifact_path TEXT NOT NULL,
            scaler_artifact_path TEXT NOT NULL,
            vectorizer_vocab_size INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT UNIQUE NOT NULL,
            text_description TEXT,
            numerical_features_json TEXT,
            categorical_features_json TEXT,
            feature_vector_json TEXT,
            feature_version_id INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE preference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_i_id TEXT NOT NULL,
            item_j_id TEXT NOT NULL,
            winner_id TEXT NOT NULL,
            query_context TEXT NOT NULL,
            position_i INTEGER,
            position_j INTEGER,
            source TEXT NOT NULL,
            user_hash TEXT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_path TEXT NOT NULL,
            feature_version_id INTEGER NOT NULL,
            training_pair_count INTEGER NOT NULL,
            ndcg_at_10 REAL,
            map_score REAL,
            pairwise_accuracy REAL,
            is_bias_corrected INTEGER NOT NULL DEFAULT 0,
            trained_at TEXT NOT NULL DEFAULT (datetime('now')),
            is_active INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    return mock_config

def test_should_retrain_no_active_model(test_db):
    res = should_retrain(test_db)
    assert res["should_retrain"] is True
    assert res["reason"] == "no_active_model"
    
def test_should_retrain_under_100_pairs(test_db):
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path) VALUES ('a', 'b')")
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, is_active, trained_at)
        VALUES ('path', 1, 10, 1, '2020-01-01 00:00:00')
    """)
    # Insert 15 pairs, which > 10 (retrain_pair_threshold) but < 100
    for i in range(15):
        cursor.execute("""
            INSERT INTO preference_log (item_i_id, item_j_id, winner_id, query_context, source, timestamp)
            VALUES (?, ?, ?, 'ctx', 'simulated', '2020-02-01 00:00:00')
        """, (f"i{i}", f"j{i}", f"i{i}"))
    conn.commit()
    conn.close()
    
    res = should_retrain(test_db)
    assert res["should_retrain"] is True
    assert res["reason"] == "pair_threshold"
    assert res["current_pair_count"] == 15
    assert res["recent_ndcg"] is None

@patch('models.retrain_trigger.compute_ndcg_at_10')
@patch('models.retrain_trigger.load_active_model')
def test_should_retrain_both_conditions(mock_load, mock_ndcg, test_db):
    mock_ndcg.return_value = 0.50 # Below 0.60 floor
    
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path) VALUES ('a', 'b')")
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, is_active, trained_at)
        VALUES ('path', 1, 100, 1, '2020-01-01 00:00:00')
    """)
    # Insert 101 pairs (so >= 100) after trained_at
    for i in range(101):
        cursor.execute("""
            INSERT INTO items (item_id, feature_version_id, feature_vector_json) VALUES (?, 1, '[0.1, 0.2]')
        """, (f"i{i}",))
        cursor.execute("""
            INSERT INTO items (item_id, feature_version_id, feature_vector_json) VALUES (?, 1, '[0.2, 0.1]')
        """, (f"j{i}",))
        cursor.execute("""
            INSERT INTO preference_log (item_i_id, item_j_id, winner_id, query_context, source, timestamp)
            VALUES (?, ?, ?, 'ctx', 'simulated', '2020-02-01 00:00:00')
        """, (f"i{i}", f"j{i}", f"i{i}"))
    conn.commit()
    conn.close()
    
    res = should_retrain(test_db)
    assert res["should_retrain"] is True
    assert res["reason"] == "both"
    assert res["current_pair_count"] == 101
    assert res["recent_ndcg"] == 0.50

@patch('models.retrain_orchestrator.run_training_pipeline_corrected')
@patch('models.retrain_orchestrator.should_retrain')
def test_execute_retraining_force(mock_should_retrain, mock_train, test_db):
    mock_should_retrain.return_value = {"should_retrain": False}
    
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path) VALUES ('a', 'b')")
    # Old model
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, ndcg_at_10, pairwise_accuracy, is_active)
        VALUES ('path', 1, 10, 0.65, 0.65, 1)
    """)
    conn.commit()
    conn.close()
    
    def side_effect(*args, **kwargs):
        # Simulate training that creates a BETTER model
        c = sqlite3.connect(test_db.db_path)
        c.execute("UPDATE model_versions SET is_active=0")
        c.execute("""
            INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, ndcg_at_10, pairwise_accuracy, is_active)
            VALUES ('new_path', 1, 20, 0.70, 0.70, 1)
        """)
        c.commit()
        last_id = c.execute("SELECT MAX(id) FROM model_versions").fetchone()[0]
        c.close()
        return last_id
        
    mock_train.side_effect = side_effect
    
    res = execute_retraining(test_db, force=True)
    assert res["retrained"] is True
    assert res["activated"] is True
    assert res["reason"] == "forced"

@patch('models.retrain_orchestrator.run_training_pipeline_corrected')
@patch('models.retrain_orchestrator.should_retrain')
def test_execute_retraining_worse_model_not_activated(mock_should_retrain, mock_train, test_db):
    mock_should_retrain.return_value = {"should_retrain": True, "reason": "test"}
    
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path) VALUES ('a', 'b')")
    # Old model
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, ndcg_at_10, pairwise_accuracy, is_active)
        VALUES ('path', 1, 10, 0.80, 0.80, 1)
    """)
    old_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    def side_effect(*args, **kwargs):
        # Simulate training that creates a WORSE model
        c = sqlite3.connect(test_db.db_path)
        c.execute("UPDATE model_versions SET is_active=0")
        c.execute("""
            INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, ndcg_at_10, pairwise_accuracy, is_active)
            VALUES ('new_path', 1, 20, 0.50, 0.50, 1)
        """)
        c.commit()
        last_id = c.execute("SELECT MAX(id) FROM model_versions").fetchone()[0]
        c.close()
        return last_id
        
    mock_train.side_effect = side_effect
    
    res = execute_retraining(test_db, force=False)
    assert res["retrained"] is True
    assert res["activated"] is False
    
    # Check DB state to ensure rollback happened
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT is_active FROM model_versions WHERE id=?", (old_id,))
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT is_active FROM model_versions WHERE id=?", (res["new_model_version_id"],))
    assert cursor.fetchone()[0] == 0
    conn.close()
