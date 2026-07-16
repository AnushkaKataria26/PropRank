import pytest
import sqlite3
import numpy as np
import torch
import os
import json
from unittest.mock import patch, mock_open
from torch.utils.data import DataLoader

from models.propensity import compute_propensity_weights
from models.train import train_ranknet, PairwiseDataset
from models.bias_diagnostics import measure_position_rank_correlation
from config.loader import get_config
from models.data_loader import load_training_pairs
from models.split import split_pairs

def get_test_config():
    config = get_test_config_original()
    config.train_epochs = 20 # Increased to ensure learning
    config.learning_rate = 0.01 # Increased to ensure divergence
    config.propensity_weight_clip = 10.0
    config.propensity_normalize = True
    return config

def get_test_config_original():
    return get_config()

def test_propensity_fallback():
    config = get_test_config()
    pairs = [
        {'label': 1.0, 'position_i': None, 'position_j': 2},
        {'label': 0.0, 'position_i': 1, 'position_j': None}
    ]
    weights, stats = compute_propensity_weights(pairs, config)
    assert stats['fallback_count'] == 2
    assert weights[0] == 1.0
    assert weights[1] == 1.0

def test_propensity_clipping():
    config = get_test_config()
    config.propensity_weight_clip = 2.0
    config.propensity_normalize = False # disable normalization to check raw values
    
    # 1/sqrt(100) = 0.1 => 1/0.1 = 10.0, which > 2.0
    pairs = [{'label': 1.0, 'position_i': 100, 'position_j': 1}]
    weights, stats = compute_propensity_weights(pairs, config)
    
    assert stats['clip_count'] == 1
    assert weights[0] == 2.0

def test_propensity_normalization():
    config = get_test_config()
    config.propensity_normalize = True
    config.propensity_weight_clip = 100.0
    
    pairs = [
        {'label': 1.0, 'position_i': 1, 'position_j': 2}, # weight 1.0
        {'label': 1.0, 'position_i': 4, 'position_j': 2}  # weight 2.0
    ]
    weights, stats = compute_propensity_weights(pairs, config)
    
    # raw mean = 1.5, normalized = 1/1.5 = 0.666, 2/1.5 = 1.333
    assert np.isclose(np.mean(weights), 1.0)
    assert np.isclose(weights[0], 1.0 / 1.5)
    assert np.isclose(weights[1], 2.0 / 1.5)

def test_weight_mismatch_exception():
    config = get_test_config()
    pairs = [{'item_i_features': np.array([1.0]), 'item_j_features': np.array([1.0]), 'label': 1.0}]
    weights = np.array([1.0, 2.0]) # mismatch
    
    with pytest.raises(ValueError, match="Length mismatch"):
        train_ranknet(pairs, config, propensity_weights=weights)

def test_weight_alignment_after_shuffle():
    pairs = []
    weights = []
    for i in range(10):
        pairs.append({
            'item_i_features': np.array([float(i)]),
            'item_j_features': np.array([float(i)]),
            'label': 1.0
        })
        weights.append(float(i * 10))
        
    dataset = PairwiseDataset(pairs, weights=weights)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in dataloader:
        feat_i = batch['item_i_features']
        weight = batch['weight']
        for f, w in zip(feat_i, weight):
            assert w.item() == f[0].item() * 10.0

def test_missing_ground_truth_exception():
    config = get_test_config()
    with patch('os.path.exists', return_value=False):
        with pytest.raises(FileNotFoundError, match="ground_truth_relevance.json is missing"):
            measure_position_rank_correlation(None, [], config)

def test_end_to_end_bias_correction():
    config = get_test_config()
    
    # Generate synthetic biased data
    np.random.seed(42)
    torch.manual_seed(42)
    
    pairs = []
    ground_truth = {}
    
    for i in range(1000):
        ctx = "query_1"
        i_id = f"item_{np.random.randint(0, 200)}"
        j_id = f"item_{np.random.randint(0, 200)}"
        
        # position bias: items at lower positions are more likely to win
        pos_i = np.random.randint(1, 20)
        pos_j = np.random.randint(1, 20)
        
        rel_i = np.random.rand()
        rel_j = np.random.rand()
        
        # Spurious feature perfectly correlated with position (lower position = higher feature)
        spur_i = 1.0 / pos_i
        spur_j = 1.0 / pos_j
        
        ground_truth[f"{i_id}|{ctx}"] = rel_i
        ground_truth[f"{j_id}|{ctx}"] = rel_j
        
        # Biased score computation
        prob_i = 1 / np.sqrt(pos_i)
        prob_j = 1 / np.sqrt(pos_j)
        score_i = rel_i * prob_i + np.random.normal(0, 0.1)
        score_j = rel_j * prob_j + np.random.normal(0, 0.1)
        
        label = 1.0 if score_i > score_j else 0.0
        
        pairs.append({
            'item_i_id': i_id,
            'item_j_id': j_id,
            'item_i_features': np.array([rel_i, spur_i]),
            'item_j_features': np.array([rel_j, spur_j]),
            'label': label,
            'query_context': ctx,
            'position_i': pos_i,
            'position_j': pos_j
        })
        
    train_pairs = pairs[:800]
    held_out_pairs = pairs[800:]
    
    # Train Baseline
    baseline_model, _ = train_ranknet(train_pairs, config, propensity_weights=None)
    
    # Train Corrected
    weights, _ = compute_propensity_weights(train_pairs, config)
    print(f"Propensity weights standard deviation: {np.std(weights)}")
    print(f"Propensity weights min: {np.min(weights)}, max: {np.max(weights)}")
    
    corrected_model, _ = train_ranknet(train_pairs, config, propensity_weights=weights)
    
    # Evaluate with mocked ground truth
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=json.dumps(ground_truth))):
             
        b_pos_corr, b_rel_corr = measure_position_rank_correlation(baseline_model, held_out_pairs, config)
        c_pos_corr, c_rel_corr = measure_position_rank_correlation(corrected_model, held_out_pairs, config)
        
    # Baseline model should have a stronger correlation with position than Corrected model
    # Note: smaller positions (e.g. 1, 2) have higher prob of winning -> higher model scores.
    # Therefore, position correlation is expected to be negative.
    # IPW correction should make this closer to 0 (i.e. less negative).
    print(f"\nb_pos: {b_pos_corr}, c_pos: {c_pos_corr}")
    print(f"b_rel: {b_rel_corr}, c_rel: {c_rel_corr}\n")
    # Check that magnitude of position correlation is reduced
    assert abs(c_pos_corr) < abs(b_pos_corr), f"IPW failed to reduce position bias: Corrected abs({c_pos_corr:.4f}) >= Baseline abs({b_pos_corr:.4f})"
    
    # Relevance correlation should be comparable or higher
    assert c_rel_corr >= b_rel_corr - 0.05, f"Corrected model lost relevance signal. c_rel: {c_rel_corr}, b_rel: {b_rel_corr}"
