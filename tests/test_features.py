import os
import sqlite3
import pandas as pd
import pytest
import numpy as np
import tempfile
import joblib

from features.ingest import load_items_csv
from features.text_features import fit_tfidf
from features.tabular_features import fit_numerical, fit_categorical
from features.build_features import build_feature_vectors
from features.persist import save_feature_version
from features.store import persist_item_features
from config.loader import Config
from db.init_db import init_db
import scipy.sparse

@pytest.fixture
def test_config():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config_dict = {
        "tfidf_max_features": 10,
        "train_batch_size": 32,
        "train_epochs": 1,
        "retrain_pair_threshold": 10,
        "confidence_pair_threshold": 5,
        "ndcg_retrain_floor": 0.5,
        "held_out_split_ratio": 0.1,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": db_path,
        "random_seed": 42
    }
    config_obj = Config(config_dict)
    
    import config.loader
    config.loader._CONFIG = config_obj
    init_db()
    
    yield config_obj
    
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "item_id": ["id1", "id2", "id3"],
        "text_description": ["nice house", "big garden", "small flat"],
        "price": [100.0, 200.0, 150.0],
        "category": ["A", "B", "A"]
    })
    path = tmp_path / "items.csv"
    df.to_csv(path, index=False)
    return str(path)

def test_ingest_duplicate_item_id(tmp_path):
    df = pd.DataFrame({"item_id": ["id1", "id1"], "text_description": ["a", "b"]})
    path = tmp_path / "dup.csv"
    df.to_csv(path, index=False)
    
    with pytest.raises(ValueError, match="Duplicate 'item_id's found"):
        load_items_csv(str(path), [], [])

def test_ingest_missing_columns(sample_csv):
    with pytest.raises(ValueError, match="Missing numerical column: 'wrong_num'"):
        load_items_csv(sample_csv, ["wrong_num"], [])
        
    with pytest.raises(ValueError, match="Missing categorical column: 'wrong_cat'"):
        load_items_csv(sample_csv, [], ["wrong_cat"])

def test_empty_text_fallback():
    df = pd.DataFrame({"item_id": ["id1", "id2"], "text_description": ["", ""]})
    vectorizer, matrix = fit_tfidf(df["text_description"], max_features=10)
    assert matrix.shape == (2, 0)
    
def test_nan_in_numerical():
    df = pd.DataFrame({"item_id": ["id1"], "price": [float("nan")]})
    
    with pytest.raises(ValueError, match="NaN found in numerical columns"):
        fit_numerical(df, ["price"])
        
    scaler, m_zero = fit_numerical(df, ["price"], fillna_strategy='zero')
    assert m_zero[0, 0] == 0.0
    
def test_zero_tabular_columns(test_config):
    df = pd.DataFrame({"item_id": ["id1"], "text_description": ["house"]})
    matrix, _, _, _ = build_feature_vectors(df, [], [], test_config)
    assert matrix.shape[1] > 0
    assert matrix.shape[0] == 1

def test_persist_updates_not_duplicates(test_config, sample_csv):
    df = load_items_csv(sample_csv, ["price"], ["category"])
    matrix, vec, scaler, enc = build_feature_vectors(df, ["price"], ["category"], test_config)
    
    vid = save_feature_version(vec, scaler, enc, test_config)
    
    persist_item_features(df, matrix, vid, test_config)
    
    conn = sqlite3.connect(test_config.db_path)
    c = conn.cursor()
    c.execute("SELECT count(*) FROM items")
    assert c.fetchone()[0] == 3
    
    df_new = df.copy()
    df_new["text_description"] = "updated"
    persist_item_features(df_new, matrix, vid, test_config)
    
    c.execute("SELECT count(*) FROM items")
    assert c.fetchone()[0] == 3
    c.execute("SELECT text_description FROM items WHERE item_id='id1'")
    assert c.fetchone()[0] == "updated"
    conn.close()

def test_roundtrip_artifact(test_config, sample_csv):
    df = load_items_csv(sample_csv, ["price"], ["category"])
    matrix, vec, scaler, enc = build_feature_vectors(df, ["price"], ["category"], test_config)
    
    vid = save_feature_version(vec, scaler, enc, test_config)
    
    conn = sqlite3.connect(test_config.db_path)
    c = conn.cursor()
    c.execute("SELECT tfidf_artifact_path, scaler_artifact_path FROM feature_versions WHERE id=?", (vid,))
    tfidf_path, scaler_path = c.fetchone()
    conn.close()
    
    loaded_vec = joblib.load(tfidf_path)
    loaded_scaler, loaded_enc = joblib.load(scaler_path)
    
    tfidf_sparse = loaded_vec.transform(df["text_description"])
    if scipy.sparse.issparse(tfidf_sparse):
        tfidf_mat = tfidf_sparse.toarray()
    else:
        tfidf_mat = tfidf_sparse
    
    num_mat = loaded_scaler.transform(df[["price"]])
    cat_mat = loaded_enc.transform(df[["category"]].astype(str))
    
    new_matrix = np.hstack([tfidf_mat, num_mat, cat_mat])
    np.testing.assert_array_almost_equal(matrix, new_matrix)

def test_unseen_categorical_inference(sample_csv):
    df = pd.DataFrame({"item_id": ["id1", "id2"], "category": ["A", "B"]})
    enc, matrix = fit_categorical(df, ["category"])
    
    df_infer = pd.DataFrame({"item_id": ["id3"], "category": ["C"]})
    infer_matrix = enc.transform(df_infer[["category"]].astype(str))
    
    assert infer_matrix.shape[1] == matrix.shape[1]
    assert np.all(infer_matrix == 0)
