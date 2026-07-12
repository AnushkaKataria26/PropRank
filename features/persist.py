import os
import joblib
import sqlite3
import time

def save_feature_version(vectorizer, scaler, encoder, config):
    """
    Serializes transformers and inserts a registry row in feature_versions.
    """
    artifacts_dir = "features/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    tfidf_path = os.path.join(artifacts_dir, f"tfidf_vectorizer_{timestamp}.joblib")
    scaler_encoder_path = os.path.join(artifacts_dir, f"scaler_encoder_{timestamp}.joblib")
    
    if os.path.exists(tfidf_path) or os.path.exists(scaler_encoder_path):
        raise FileExistsError("Artifact filename collision detected.")
        
    joblib.dump(vectorizer, tfidf_path)
    joblib.dump((scaler, encoder), scaler_encoder_path)
    
    try:
        vocab_size = len(vectorizer.vocabulary_)
    except AttributeError:
        vocab_size = 0
        
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path, vectorizer_vocab_size)
        VALUES (?, ?, ?)
    """, (tfidf_path, scaler_encoder_path, vocab_size))
    
    feature_version_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return feature_version_id
