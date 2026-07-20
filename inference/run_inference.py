import logging
from inference.model_loader import load_active_model
from inference.candidates import get_candidates_for_context
from inference.confidence import check_context_confidence
from inference.bm25_fallback import bm25_rank
from inference.score_items import score_with_model
from inference.cache import get_cached_result, set_cached_result, _CACHE

logger = logging.getLogger(__name__)

def run_inference(query_context, top_k, config, cache=None):
    """
    End-to-end inference pipeline serving rankings for a query context.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be strictly positive, got {top_k}")
        
    if cache is None:
        cache = _CACHE
        
    # 1. Load Active Model
    bundle = load_active_model(config)
    
    # 2. Get Candidates
    candidates = get_candidates_for_context(query_context, bundle.feature_version_id, config)
    candidate_item_ids = [c['item_id'] for c in candidates]
    
    # 3. Check Cache
    cached_result = get_cached_result(cache, query_context, candidate_item_ids, bundle.model_version_id)
    if cached_result is not None:
        logger.info(f"Cache hit for query_context='{query_context}'")
        # Apply top_k slicing in case the cached result is larger
        # (Assuming the cached result was the full ranking)
        return cached_result[:top_k]
        
    # 4. Check Confidence
    confidence_flag = check_context_confidence(query_context, config)
    fallback_used = (confidence_flag == 'FALLBACK')
    
    if fallback_used:
        logger.info(f"Confidence below threshold. Using BM25 fallback for '{query_context}'.")
        ranked_items = bm25_rank(candidates, query_context, config)
    else:
        logger.info(f"Confidence above threshold. Using model scoring for '{query_context}'.")
        ranked_items = score_with_model(bundle, candidates, config)
        
    # 5. Assemble Output
    final_output = []
    for rank, (item_id, score) in enumerate(ranked_items, start=1):
        final_output.append({
            'item_id': item_id,
            'rank': rank,
            'score': score,
            'confidence_flag': confidence_flag,
            'fallback_used': fallback_used
        })
        
    # 6. Store full result in cache
    set_cached_result(cache, query_context, candidate_item_ids, bundle.model_version_id, final_output)
    
    if top_k > len(final_output):
        logger.info(f"Requested top_k={top_k} exceeds available candidates ({len(final_output)}). Returning all.")
        
    return final_output[:top_k]
