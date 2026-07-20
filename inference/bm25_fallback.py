import re
import logging
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def tokenize(text):
    """
    Simple tokenizer that lowercases and splits on non-alphanumeric characters.
    Note: This is intentionally simple and not a full NLP tokenizer, consistent 
    with the project's local-only, dependency-light scope.
    """
    if not text:
        return []
    return [t for t in re.split(r'[^a-z0-9]+', text.lower()) if t]

def bm25_rank(candidate_items, query_context, config):
    """
    Ranks candidate items using BM25Okapi over the text_description field.
    
    candidate_items: list of dicts with 'item_id' and 'text_description'
    query_context: str, the search query
    config: configuration object with bm25_k1 and bm25_b attributes
    
    Returns: list of (item_id, bm25_score) sorted descending by score.
    """
    if not candidate_items:
        raise ValueError("candidate_items cannot be empty at inference time.")
        
    query_tokens = tokenize(query_context)
    
    if not query_tokens:
        logger.warning(
            "BM25 query_context tokenized to zero terms. "
            "Falling back to deterministic tie-breaker ranking (alphabetical by item_id)."
        )
        # Deterministic alphabetical fallback
        sorted_candidates = sorted(candidate_items, key=lambda x: str(x['item_id']))
        return [(item['item_id'], 0.0) for item in sorted_candidates]
        
    corpus_tokens = []
    for item in candidate_items:
        corpus_tokens.append(tokenize(item.get('text_description', '')))
        
    bm25 = BM25Okapi(corpus_tokens, k1=config.bm25_k1, b=config.bm25_b)
    scores = bm25.get_scores(query_tokens)
    
    scored_items = [
        (item['item_id'], float(score)) 
        for item, score in zip(candidate_items, scores)
    ]
    
    # Sort descending by score. Break ties alphabetically by item_id
    scored_items.sort(key=lambda x: (-x[1], str(x[0])))
    
    return scored_items
