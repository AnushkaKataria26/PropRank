import pytest
import sqlite3
import numpy as np
import torch
import json
import os
from unittest.mock import patch, MagicMock

from config.loader import Config
from inference.bm25_fallback import bm25_rank
from inference.model_loader import load_active_model, ModelBundle
from inference.candidates import get_candidates_for_context
from inference.confidence import check_context_confidence
from inference.score_items import score_with_model
from inference.cache import get_cached_result, set_cached_result
from inference.run_inference import run_inference
from models.ranknet import RankNetMLP

@pytest.fixture
def test_config(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    config_dict = {
        "tfidf_max_features": 100,
        "train_batch_size": 32,
        "train_epochs": 1,
        "retrain_pair_threshold": 100,
        "confidence_pair_threshold": 5,
        "ndcg_retrain_floor": 0.8,
        "held_out_split_ratio": 0.2,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": str(db_path),
        "random_seed": 42
    }
    return Config(config_dict)

@pytest.fixture
def setup_db(test_config):
    conn = sqlite3.connect(test_config.db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE feature_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, tfidf_artifact_path TEXT, scaler_artifact_path TEXT)")
    cursor.execute("CREATE TABLE items (item_id TEXT, text_description TEXT, feature_vector_json TEXT, feature_version_id INTEGER)")
    cursor.execute("CREATE TABLE preference_log (query_context TEXT)")
    cursor.execute("CREATE TABLE model_versions (id INTEGER PRIMARY KEY AUTOINCREMENT, artifact_path TEXT, feature_version_id INTEGER, is_active INTEGER)")
    conn.commit()
    conn.close()

def test_bm25_empty_candidates(test_config):
    with pytest.raises(ValueError, match="cannot be empty"):
        bm25_rank([], "query", test_config)

def test_bm25_zero_terms_fallback(test_config, caplog):
    candidates = [
        {'item_id': 'B', 'text_description': 'b'},
        {'item_id': 'A', 'text_description': 'a'}
    ]
    # Empty string or punctuation will tokenize to 0 terms
    ranked = bm25_rank(candidates, "!!!", test_config)
    assert len(ranked) == 2
    assert ranked[0] == ('A', 0.0)
    assert ranked[1] == ('B', 0.0)
    assert "tokenized to zero terms" in caplog.text

def test_load_active_model_zero_rows(setup_db, test_config):
    with pytest.raises(ValueError, match="No active model found"):
        load_active_model(test_config)

def test_load_active_model_multiple_rows(setup_db, test_config):
    conn = sqlite3.connect(test_config.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO model_versions (is_active) VALUES (1), (1)")
    conn.commit()
    conn.close()
    
    with pytest.raises(RuntimeError, match="Database corruption"):
        load_active_model(test_config)

@patch('inference.model_loader.joblib.load')
@patch('inference.model_loader.torch.load')
def test_load_active_model_success(mock_torch, mock_joblib, setup_db, test_config):
    conn = sqlite3.connect(test_config.db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO feature_versions (tfidf_artifact_path, scaler_artifact_path) VALUES ('dummy_tfidf', 'dummy_scaler')")
    fv_id = cursor.lastrowid
    cursor.execute("INSERT INTO model_versions (artifact_path, feature_version_id, is_active) VALUES ('dummy_model', ?, 1)", (fv_id,))
    mv_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    mock_joblib.side_effect = ["vectorizer", ("scaler", "encoder")]
    
    # Create a real state_dict for RankNetMLP(2)
    model = RankNetMLP(2)
    mock_torch.return_value = model.state_dict()
    
    bundle = load_active_model(test_config)
    
    assert bundle.model_version_id == mv_id
    assert bundle.feature_version_id == fv_id
    assert bundle.vectorizer == "vectorizer"
    assert bundle.scaler == "scaler"
    assert not bundle.model.training # Should be eval mode
    assert bundle.model.mlp[0].in_features == 2

def test_get_candidates_empty(setup_db, test_config):
    with pytest.raises(ValueError, match="No items found"):
        get_candidates_for_context("ctx", 999, test_config)

def test_check_confidence(setup_db, test_config):
    conn = sqlite3.connect(test_config.db_path)
    cursor = conn.cursor()
    for _ in range(4):
        cursor.execute("INSERT INTO preference_log (query_context) VALUES ('fallback_ctx')")
    for _ in range(5):
        cursor.execute("INSERT INTO preference_log (query_context) VALUES ('confident_ctx')")
    conn.commit()
    conn.close()
    
    assert check_context_confidence("fallback_ctx", test_config) == 'FALLBACK'
    assert check_context_confidence("confident_ctx", test_config) == 'CONFIDENT'

def test_score_with_model_nan():
    model = RankNetMLP(2)
    bundle = ModelBundle(model, 1, None, None, None, 1)
    cands = [{'item_id': 'X', 'feature_vector_json': '[1.0, NaN]'}]
    with pytest.raises(ValueError, match="NaN detected"):
        score_with_model(bundle, cands, None)

def test_score_with_model_dim_mismatch():
    model = RankNetMLP(2)
    bundle = ModelBundle(model, 1, None, None, None, 1)
    cands = [{'item_id': 'X', 'feature_vector_json': '[1.0]'}]
    with pytest.raises(ValueError, match="Dimension mismatch"):
        score_with_model(bundle, cands, None)

def test_cache_logic():
    cache = {}
    set_cached_result(cache, "q", ["2", "1"], 1, "result1")
    # Same args but order changed should hit cache
    res = get_cached_result(cache, "q", ["1", "2"], 1)
    assert res == "result1"
    
    # Model version changed -> miss
    res2 = get_cached_result(cache, "q", ["1", "2"], 2)
    assert res2 is None

@patch('inference.run_inference.load_active_model')
@patch('inference.run_inference.get_candidates_for_context')
@patch('inference.run_inference.check_context_confidence')
@patch('inference.run_inference.score_with_model')
@patch('inference.run_inference.bm25_rank')
def test_run_inference_topk(mock_bm25, mock_score, mock_conf, mock_cands, mock_load, test_config):
    mock_conf.return_value = 'CONFIDENT'
    mock_cands.return_value = [{'item_id': '1'}, {'item_id': '2'}]
    mock_score.return_value = [('1', 1.0), ('2', 0.5)]
    mock_load.return_value = ModelBundle(None, 1, None, None, None, 1)
    
    with pytest.raises(ValueError, match="strictly positive"):
        run_inference("q", 0, test_config, cache={})
        
    res = run_inference("q", 5, test_config, cache={})
    assert len(res) == 2
    assert res[0]['item_id'] == '1'

@patch('inference.run_inference.load_active_model')
@patch('inference.run_inference.get_candidates_for_context')
@patch('inference.run_inference.check_context_confidence')
@patch('inference.run_inference.score_with_model')
@patch('inference.run_inference.bm25_rank')
def test_run_inference_e2e(mock_bm25, mock_score, mock_conf, mock_cands, mock_load, test_config):
    bundle = ModelBundle(None, 1, None, None, None, 1)
    mock_load.return_value = bundle
    cands = [{'item_id': '1'}, {'item_id': '2'}]
    mock_cands.return_value = cands
    
    cache = {}
    
    # 1. Fallback
    mock_conf.return_value = 'FALLBACK'
    mock_bm25.return_value = [('2', 2.0), ('1', 1.0)]
    
    res1 = run_inference("ctx1", 2, test_config, cache=cache)
    assert len(res1) == 2
    assert res1[0]['item_id'] == '2'
    assert res1[0]['confidence_flag'] == 'FALLBACK'
    assert res1[0]['fallback_used'] is True
    assert res1[0]['rank'] == 1
    
    # 2. Confident
    mock_conf.return_value = 'CONFIDENT'
    mock_score.return_value = [('1', 5.0), ('2', -1.0)]
    
    res2 = run_inference("ctx2", 1, test_config, cache=cache)
    assert len(res2) == 1
    assert res2[0]['item_id'] == '1'
    assert res2[0]['confidence_flag'] == 'CONFIDENT'
    assert res2[0]['fallback_used'] is False
    assert res2[0]['rank'] == 1
    
    # 3. Cache hit on ctx2
    mock_score.reset_mock()
    res3 = run_inference("ctx2", 1, test_config, cache=cache)
    assert len(res3) == 1
    assert res3[0]['item_id'] == '1'
    mock_score.assert_not_called() # Should have hit cache
