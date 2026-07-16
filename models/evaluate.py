import torch
import numpy as np
import logging
from collections import defaultdict
from sklearn.metrics import ndcg_score, average_precision_score

logger = logging.getLogger(__name__)

def _get_scores(model, held_out_pairs):
    if not held_out_pairs:
        raise ValueError("held_out_pairs cannot be empty")
        
    model.eval()
    feat_i_list = [torch.tensor(p['item_i_features'], dtype=torch.float32) for p in held_out_pairs]
    feat_j_list = [torch.tensor(p['item_j_features'], dtype=torch.float32) for p in held_out_pairs]
    
    with torch.no_grad():
        feat_i_tensor = torch.stack(feat_i_list)
        feat_j_tensor = torch.stack(feat_j_list)
        
        score_i = model(feat_i_tensor).squeeze(-1).numpy()
        score_j = model(feat_j_tensor).squeeze(-1).numpy()
        
    return score_i, score_j

def compute_pairwise_accuracy(model, held_out_pairs):
    score_i, score_j = _get_scores(model, held_out_pairs)
    labels = np.array([p['label'] for p in held_out_pairs])
    
    diff = score_i - score_j
    ties = (diff == 0)
    num_ties = np.sum(ties)
    
    correct = ((diff > 0) & (labels == 1.0)) | ((diff < 0) & (labels == 0.0))
    total = len(held_out_pairs)
    num_correct = np.sum(correct)
    
    accuracy = num_correct / total
    tie_rate = num_ties / total
    
    logger.info(f"Pairwise Accuracy: {accuracy:.4f}, Tie Rate: {tie_rate:.4f}")
    return accuracy, tie_rate

def _get_graded_relevance_per_context(model, held_out_pairs):
    if not held_out_pairs:
        raise ValueError("held_out_pairs cannot be empty")
        
    context_to_wins = defaultdict(lambda: defaultdict(int))
    context_to_items = defaultdict(set)
    item_features = {}
    
    for p in held_out_pairs:
        ctx = p['query_context']
        i_id = p['item_i_id']
        j_id = p['item_j_id']
        
        context_to_items[ctx].add(i_id)
        context_to_items[ctx].add(j_id)
        item_features[i_id] = p['item_i_features']
        item_features[j_id] = p['item_j_features']
        
        if p['label'] == 1.0:
            context_to_wins[ctx][i_id] += 1
        else:
            context_to_wins[ctx][j_id] += 1
            
    contexts = []
    y_true_all = []
    y_pred_all = []
    
    model.eval()
    
    for ctx, items in context_to_items.items():
        if len(items) < 2:
            logger.info(f"Skipping context {ctx} for NDCG/MAP as it has < 2 items.")
            continue
            
        items_list = list(items)
        y_true = np.array([context_to_wins[ctx][item_id] for item_id in items_list])
        
        feats = [torch.tensor(item_features[item_id], dtype=torch.float32) for item_id in items_list]
        with torch.no_grad():
            feats_tensor = torch.stack(feats)
            y_pred = model(feats_tensor).squeeze(-1).numpy()
            
        y_true_all.append([y_true])
        y_pred_all.append([y_pred])
        contexts.append(ctx)
        
    return y_true_all, y_pred_all, contexts

def compute_ndcg_at_10(model, held_out_pairs):
    y_true_all, y_pred_all, contexts = _get_graded_relevance_per_context(model, held_out_pairs)
    if not y_true_all:
        logger.warning("No valid contexts with >= 2 items for NDCG.")
        return 0.0
        
    ndcg_scores = []
    for y_t, y_p in zip(y_true_all, y_pred_all):
        score = ndcg_score(y_t, y_p, k=10)
        ndcg_scores.append(score)
        
    return np.mean(ndcg_scores)

def compute_map(model, held_out_pairs):
    y_true_all, y_pred_all, contexts = _get_graded_relevance_per_context(model, held_out_pairs)
    if not y_true_all:
        logger.warning("No valid contexts with >= 2 items for MAP.")
        return 0.0
        
    map_scores = []
    for y_t, y_p in zip(y_true_all, y_pred_all):
        y_t_bin = (y_t[0] > 0).astype(int)
        if np.sum(y_t_bin) == 0:
            map_scores.append(0.0)
            continue
            
        score = average_precision_score(y_t_bin, y_p[0])
        if np.isnan(score):
            score = 0.0
        map_scores.append(score)
        
    return np.mean(map_scores)
