import sqlite3
import torch
import logging
from models.data_loader import load_training_pairs
from models.split import split_pairs
from models.evaluate import compute_pairwise_accuracy, compute_ndcg_at_10, compute_map
from models.bias_diagnostics import measure_position_rank_correlation
from models.ranknet import RankNetMLP

logger = logging.getLogger(__name__)

def compare_baseline_vs_corrected(config):
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT artifact_path, feature_version_id FROM model_versions 
        WHERE is_bias_corrected = 0 
        ORDER BY id DESC LIMIT 1
    """)
    baseline_row = cursor.fetchone()
    if not baseline_row:
        conn.close()
        raise ValueError("Baseline model (is_bias_corrected=0) is missing from model_versions.")
        
    cursor.execute("""
        SELECT artifact_path, feature_version_id FROM model_versions 
        WHERE is_bias_corrected = 1 
        ORDER BY id DESC LIMIT 1
    """)
    corrected_row = cursor.fetchone()
    if not corrected_row:
        conn.close()
        raise ValueError("Corrected model (is_bias_corrected=1) is missing from model_versions.")
        
    conn.close()
    
    feat_ver = baseline_row['feature_version_id']
    
    # Load identical split using the same config
    logger.info("Loading identical held-out split for comparison...")
    pairs = load_training_pairs(feat_ver, query_contexts=None, config=config)
    _, held_out_pairs = split_pairs(pairs, config)
    
    input_dim = held_out_pairs[0]['item_i_features'].shape[0]
    
    logger.info(f"Evaluating Baseline Model from {baseline_row['artifact_path']}")
    baseline_model = RankNetMLP(input_dim)
    baseline_model.load_state_dict(torch.load(baseline_row['artifact_path']))
    baseline_model.eval()
    
    logger.info(f"Evaluating Corrected Model from {corrected_row['artifact_path']}")
    corrected_model = RankNetMLP(input_dim)
    corrected_model.load_state_dict(torch.load(corrected_row['artifact_path']))
    corrected_model.eval()
    
    b_acc, _ = compute_pairwise_accuracy(baseline_model, held_out_pairs)
    b_ndcg = compute_ndcg_at_10(baseline_model, held_out_pairs)
    b_map = compute_map(baseline_model, held_out_pairs)
    b_pos_corr, b_rel_corr = measure_position_rank_correlation(baseline_model, held_out_pairs, config)
    
    c_acc, _ = compute_pairwise_accuracy(corrected_model, held_out_pairs)
    c_ndcg = compute_ndcg_at_10(corrected_model, held_out_pairs)
    c_map = compute_map(corrected_model, held_out_pairs)
    c_pos_corr, c_rel_corr = measure_position_rank_correlation(corrected_model, held_out_pairs, config)
    
    print("\n=== Model Comparison Report ===")
    print(f"{'Metric':<30} | {'Baseline (Biased)':<20} | {'IPW Corrected':<20} | {'Delta':<10}")
    print("-" * 88)
    
    metrics = [
        ("Pairwise Accuracy", b_acc, c_acc),
        ("NDCG@10", b_ndcg, c_ndcg),
        ("MAP", b_map, c_map),
        ("Position-Score Correlation", b_pos_corr, c_pos_corr),
        ("Relevance-Score Correlation", b_rel_corr, c_rel_corr)
    ]
    
    for name, b_val, c_val in metrics:
        delta = c_val - b_val
        print(f"{name:<30} | {b_val:<20.4f} | {c_val:<20.4f} | {delta:<10.4f}")
        
    print("-" * 88)
