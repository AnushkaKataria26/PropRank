import os
import sqlite3
import datetime
import logging
import torch

from models.data_loader import load_training_pairs
from models.split import split_pairs
from models.train import train_ranknet
from models.evaluate import compute_pairwise_accuracy, compute_ndcg_at_10, compute_map

logger = logging.getLogger(__name__)

def run_training_pipeline(feature_version_id, query_contexts, config):
    logger.info("Loading training pairs...")
    loaded_pairs = load_training_pairs(feature_version_id, query_contexts, config)
    
    logger.info("Splitting pairs...")
    train_pairs, held_out_pairs = split_pairs(loaded_pairs, config)
    
    logger.info(f"Train pairs: {len(train_pairs)}, Held-out pairs: {len(held_out_pairs)}")
    
    logger.info("Training RankNet...")
    model, epoch_losses = train_ranknet(train_pairs, config, propensity_weights=None)
    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    
    logger.info("Evaluating model...")
    accuracy, tie_rate = compute_pairwise_accuracy(model, held_out_pairs)
    ndcg_10 = compute_ndcg_at_10(model, held_out_pairs)
    map_score = compute_map(model, held_out_pairs)
    
    # Save model artifact
    artifacts_dir = "models/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = os.path.join(artifacts_dir, f"ranknet_v{feature_version_id}_{timestamp}.pt")
    torch.save(model.state_dict(), artifact_path)
    logger.info(f"Saved model to {artifact_path}")
    
    # Update database
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    
    # Deactivate existing active model versions
    cursor.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
    
    # Insert new model version
    cursor.execute("""
        INSERT INTO model_versions (
            artifact_path, feature_version_id, training_pair_count,
            ndcg_at_10, map_score, pairwise_accuracy, is_bias_corrected, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, 0, 1)
    """, (
        artifact_path, feature_version_id, len(train_pairs),
        ndcg_10, map_score, accuracy
    ))
    
    model_version_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    print("\n=== Training Summary ===")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Held-out pairs: {len(held_out_pairs)}")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Pairwise Accuracy: {accuracy:.4f}")
    print(f"NDCG@10: {ndcg_10:.4f}")
    print(f"MAP: {map_score:.4f}")
    print(f"Artifact path: {artifact_path}")
    print(f"Model version ID: {model_version_id}")
    
    return model_version_id
