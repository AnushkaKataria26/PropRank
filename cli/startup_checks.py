import os
import sqlite3
import sys

def validate_environment(config):
    """
    Validates the existence of the database, its schema, and creates required artifact directories.
    """
    db_path = config.db_path
    
    # 1. Check if DB exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}.", file=sys.stderr)
        print("Please initialize the database by running: python -m db.init_db", file=sys.stderr)
        sys.exit(1)
        
    # 2. Check Schema
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
    except Exception as e:
        print(f"Error reading database at {db_path}: {e}", file=sys.stderr)
        sys.exit(1)
        
    expected_tables = {"feature_versions", "items", "preference_log", "model_versions"}
    missing_tables = expected_tables - tables
    
    if missing_tables:
        print(f"Error: Database at {db_path} is missing required tables: {missing_tables}", file=sys.stderr)
        print("Please initialize the database by running: python -m db.init_db", file=sys.stderr)
        sys.exit(1)
        
    # 3. Create Artifact Directories
    required_dirs = [
        "features/artifacts",
        "models/artifacts",
        "simulator/artifacts"
    ]
    
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
