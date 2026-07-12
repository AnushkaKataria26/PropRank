import sqlite3
import os
from config.loader import get_config

def init_db():
    config = get_config()
    db_path = config.db_path
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    db_exists = os.path.exists(db_path)
    if db_exists:
        print(f"Warning: Database at {db_path} already exists. Running IF NOT EXISTS schema creation.")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 4. feature_versions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tfidf_artifact_path TEXT NOT NULL,
        scaler_artifact_path TEXT NOT NULL,
        vectorizer_vocab_size INTEGER,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """)
    
    # 1. items
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id TEXT UNIQUE NOT NULL,
        text_description TEXT,
        numerical_features_json TEXT,
        categorical_features_json TEXT,
        feature_vector_json TEXT,
        feature_version_id INTEGER,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY(feature_version_id) REFERENCES feature_versions(id)
    )
    """)
    
    # Ensure explicit index on items(item_id)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_items_item_id ON items(item_id)")
    
    # 2. preference_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preference_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_i_id TEXT NOT NULL,
        item_j_id TEXT NOT NULL,
        winner_id TEXT NOT NULL,
        query_context TEXT NOT NULL,
        position_i INTEGER,
        position_j INTEGER,
        source TEXT NOT NULL CHECK(source IN ('simulated','real')),
        user_hash TEXT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        CHECK (winner_id = item_i_id OR winner_id = item_j_id),
        CHECK (item_i_id != item_j_id)
    )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pref_log_query_time ON preference_log(query_context, timestamp DESC)")
    
    # 3. model_versions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        artifact_path TEXT NOT NULL,
        feature_version_id INTEGER NOT NULL,
        training_pair_count INTEGER NOT NULL,
        ndcg_at_10 REAL,
        map_score REAL,
        pairwise_accuracy REAL,
        is_bias_corrected INTEGER NOT NULL DEFAULT 0,
        trained_at TEXT NOT NULL DEFAULT (datetime('now')),
        is_active INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(feature_version_id) REFERENCES feature_versions(id)
    )
    """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
