import json
import os

class Config:
    def __init__(self, config_dict):
        self.tfidf_max_features = config_dict["tfidf_max_features"]
        self.train_batch_size = config_dict["train_batch_size"]
        self.train_epochs = config_dict["train_epochs"]
        self.retrain_pair_threshold = config_dict["retrain_pair_threshold"]
        self.confidence_pair_threshold = config_dict["confidence_pair_threshold"]
        self.ndcg_retrain_floor = config_dict["ndcg_retrain_floor"]
        self.held_out_split_ratio = config_dict["held_out_split_ratio"]
        self.bm25_k1 = config_dict["bm25_k1"]
        self.bm25_b = config_dict["bm25_b"]
        self.db_path = config_dict["db_path"]
        self.random_seed = config_dict["random_seed"]
        self.learning_rate = config_dict.get("learning_rate", 0.001)
        self.propensity_weight_clip = config_dict.get("propensity_weight_clip", 10.0)
        self.propensity_normalize = config_dict.get("propensity_normalize", True)

_CONFIG = None

def load_config(config_path="config/config.json"):
    global _CONFIG
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        try:
            config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse configuration file: {e}")
            
    required_keys = [
        "tfidf_max_features", "train_batch_size", "train_epochs", 
        "retrain_pair_threshold", "confidence_pair_threshold", 
        "ndcg_retrain_floor", "held_out_split_ratio", "bm25_k1", 
        "bm25_b", "db_path", "random_seed", "learning_rate"
    ]
    
    for key in required_keys:
        if key not in config_dict:
            raise KeyError(f"Missing required configuration key: {key}")
            
    _CONFIG = Config(config_dict)
    return _CONFIG

def get_config():
    if _CONFIG is None:
        return load_config()
    return _CONFIG
