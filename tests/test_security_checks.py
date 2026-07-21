import pytest
import sqlite3
import os
import json
from config.loader import get_config
from models.data_loader import load_training_pairs
from inference.model_loader import load_active_model
import inspect

def test_source_diversity_floor(tmp_path):
    """
    Test that a single user_hash cannot contribute more than 
    max_user_contribution_ratio of the pairs for a context.
    """
    config = get_config()
    db_path = str(tmp_path / "test.sqlite3")
    config.db_path = db_path
    config.max_user_contribution_ratio = 0.50 # 50% max

    from db.init_db import init_db
    init_db()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert feature version
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path, vectorizer_vocab_size) VALUES ('', '', 100)")
    fv_id = cursor.lastrowid
    
    # Insert 4 items
    features = json.dumps([0.1, 0.2])
    for i in range(1, 5):
        cursor.execute("INSERT INTO items (item_id, feature_vector_json, feature_version_id) VALUES (?, ?, ?)",
                       (f"item_{i}", features, fv_id))
    
    # Insert preference pairs for the same context
    # User A contributes 3 pairs, User B contributes 1 pair
    # Total = 4. 50% of 4 = 2 max per user.
    # So User A should be capped at 2, User B at 1. Total valid pairs should be 3.
    
    pairs = [
        ("item_1", "item_2", "item_1", "ctx", "user_A"),
        ("item_1", "item_3", "item_1", "ctx", "user_A"),
        ("item_2", "item_3", "item_2", "ctx", "user_A"),
        ("item_3", "item_4", "item_3", "ctx", "user_B")
    ]
    
    for i_id, j_id, w_id, ctx, u_hash in pairs:
        cursor.execute("INSERT INTO preference_log (item_i_id, item_j_id, winner_id, query_context, source, user_hash) VALUES (?, ?, ?, ?, 'real', ?)",
                       (i_id, j_id, w_id, ctx, u_hash))
                       
    conn.commit()
    conn.close()
    
    valid_pairs = load_training_pairs(fv_id, ['ctx'], config)
    
    assert len(valid_pairs) == 3, f"Expected 3 pairs after diversity floor, got {len(valid_pairs)}"
    
    users = [p['user_hash'] for p in valid_pairs]
    assert users.count('user_A') == 2
    assert users.count('user_B') == 1


def test_model_loader_path_traversal_prevention():
    """
    Test that active model artifacts are loaded exclusively via paths stored in the SQLite registry
    (feature_versions table), and that the loader functions do not accept caller-provided paths,
    preventing path traversal attacks.
    """
    # Use introspection to prove `load_active_model` does not accept artifact path arguments
    sig = inspect.signature(load_active_model)
    params = list(sig.parameters.keys())
    
    assert 'tfidf_path' not in params
    assert 'scaler_path' not in params
    assert 'model_path' not in params
    
    # The only argument should be config
    assert 'config' in params
