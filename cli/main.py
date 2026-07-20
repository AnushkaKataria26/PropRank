import argparse
import sys
import json
import sqlite3
import traceback

from config.loader import get_config
from cli.startup_checks import validate_environment

def main():
    parser = argparse.ArgumentParser(prog="propRank", description="PropRank CLI for managing the learning-to-rank pipeline.")
    parser.add_argument("--debug", action="store_true", help="Print full tracebacks for errors")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. ingest
    parser_ingest = subparsers.add_parser("ingest", help="Run feature extraction pipeline")
    parser_ingest.add_argument("--csv", required=True, help="Path to input CSV file")
    parser_ingest.add_argument("--numerical", help="Comma-separated numerical feature columns")
    parser_ingest.add_argument("--categorical", help="Comma-separated categorical feature columns")
    
    # 2. simulate
    parser_simulate = subparsers.add_parser("simulate", help="Run simulation to generate preference pairs")
    parser_simulate.add_argument("--contexts", required=True, help="Comma-separated list of query contexts")
    parser_simulate.add_argument("--pairs-per-context", type=int, required=True, help="Number of pairs to simulate per context")
    
    # 3. train
    parser_train = subparsers.add_parser("train", help="Train a model on generated pairs")
    parser_train.add_argument("--feature-version", type=int, required=True, help="Feature version ID to use")
    parser_train.add_argument("--contexts", help="Comma-separated list of query contexts to train on (default: all)")
    parser_train.add_argument("--baseline-only", action="store_true", help="Train without bias correction (baseline)")
    
    # 4. rank
    parser_rank = subparsers.add_parser("rank", help="Rank candidates using the active model")
    parser_rank.add_argument("--context", required=True, help="Query context")
    parser_rank.add_argument("--top-k", type=int, required=True, help="Number of top items to return")
    
    # 5. retrain-check
    parser_retrain_check = subparsers.add_parser("retrain-check", help="Check if retraining is warranted")
    
    # 6. retrain
    parser_retrain = subparsers.add_parser("retrain", help="Execute retraining if warranted")
    parser_retrain.add_argument("--force", action="store_true", help="Force retraining even if not triggered")
    
    # 7. rollback
    parser_rollback = subparsers.add_parser("rollback", help="Rollback active model to a specific version")
    parser_rollback.add_argument("--model-version-id", type=int, required=True, help="Model version ID to rollback to")
    parser_rollback.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    
    # 8. history
    parser_history = subparsers.add_parser("history", help="List all model versions")
    
    args = parser.parse_args()
    
    try:
        config = get_config()
        validate_environment(config)
        
        if args.command == "ingest":
            import os
            if not os.path.exists(args.csv):
                print(f"Error: CSV file not found at {args.csv}", file=sys.stderr)
                sys.exit(1)
            
            from features.run_feature_pipeline import run_feature_pipeline
            num_cols = args.numerical.split(",") if args.numerical else []
            cat_cols = args.categorical.split(",") if args.categorical else []
            
            num_cols = [c.strip() for c in num_cols if c.strip()]
            cat_cols = [c.strip() for c in cat_cols if c.strip()]
            
            run_feature_pipeline(args.csv, num_cols, cat_cols, config)
            
        elif args.command == "simulate":
            from simulator.run_simulation import run_simulation
            contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
            if not contexts:
                print("Error: --contexts must contain at least one valid context.", file=sys.stderr)
                sys.exit(1)
            
            run_simulation(contexts, args.pairs_per_context, config)
            
        elif args.command == "train":
            contexts = None
            if args.contexts:
                contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
                
            if args.baseline_only:
                from models.run_training import run_training_pipeline
                run_training_pipeline(args.feature_version, contexts, config)
            else:
                from models.run_training_corrected import run_training_pipeline_corrected
                run_training_pipeline_corrected(args.feature_version, contexts, config)
                
        elif args.command == "rank":
            from inference.run_inference import run_inference
            results = run_inference(args.context, args.top_k, config)
            print(json.dumps(results, indent=2))
            
        elif args.command == "retrain-check":
            from models.retrain_trigger import should_retrain
            result = should_retrain(config)
            print(json.dumps(result, indent=2))
            
        elif args.command == "retrain":
            from models.retrain_orchestrator import execute_retraining
            result = execute_retraining(config, force=args.force)
            print(json.dumps(result, indent=2))
            
        elif args.command == "rollback":
            from models.rollback import rollback_to_version
            if not args.yes:
                ans = input(f"Are you sure you want to roll back to model version {args.model_version_id}? [y/N] ")
                if ans.lower() not in ["y", "yes"]:
                    print("Rollback cancelled.")
                    sys.exit(0)
                    
            result = rollback_to_version(args.model_version_id, config)
            print(json.dumps(result, indent=2))
            
        elif args.command == "history":
            conn = sqlite3.connect(config.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, is_bias_corrected, training_pair_count, ndcg_at_10, 
                       map_score, pairwise_accuracy, trained_at, is_active
                FROM model_versions
                ORDER BY trained_at DESC
            """)
            rows = cursor.fetchall()
            conn.close()
            
            print(f"{'ID':<5} | {'Bias Corr':<9} | {'Pairs':<6} | {'NDCG@10':<7} | {'MAP':<7} | {'Accuracy':<8} | {'Active':<6} | {'Trained At'}")
            print("-" * 85)
            for r in rows:
                ndcg = f"{r['ndcg_at_10']:.4f}" if r['ndcg_at_10'] is not None else "None"
                map_s = f"{r['map_score']:.4f}" if r['map_score'] is not None else "None"
                acc = f"{r['pairwise_accuracy']:.4f}" if r['pairwise_accuracy'] is not None else "None"
                print(f"{r['id']:<5} | {r['is_bias_corrected']:<9} | {r['training_pair_count']:<6} | {ndcg:<7} | {map_s:<7} | {acc:<8} | {r['is_active']:<6} | {r['trained_at']}")
                
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
