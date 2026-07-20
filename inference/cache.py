import hashlib
import json

# Process-local, in-memory cache
# Note: Consistent with the MVP scope, there is no persistence across restarts.
_CACHE = {}

def _generate_key(query_context, candidate_item_ids, model_version_id):
    """
    Generates a deterministic hash key for cache lookups.
    """
    sorted_ids = sorted(candidate_item_ids)
    
    key_dict = {
        'query_context': query_context,
        'candidates': sorted_ids,
        'model_version_id': model_version_id
    }
    
    key_str = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

def get_cached_result(cache, query_context, candidate_item_ids, model_version_id):
    key = _generate_key(query_context, candidate_item_ids, model_version_id)
    return cache.get(key)
    
def set_cached_result(cache, query_context, candidate_item_ids, model_version_id, result):
    key = _generate_key(query_context, candidate_item_ids, model_version_id)
    cache[key] = result
