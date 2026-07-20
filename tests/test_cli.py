import pytest
import sqlite3
import subprocess
import os

from cli.startup_checks import validate_environment
from config.loader import Config
from models.rollback import rollback_to_version

@pytest.fixture
def mock_config(tmp_path):
    db_path = tmp_path / "test_propRank.sqlite3"
    config_dict = {
        "tfidf_max_features": 100,
        "train_batch_size": 32,
        "train_epochs": 1,
        "retrain_pair_threshold": 10,
        "confidence_pair_threshold": 5,
        "ndcg_retrain_floor": 0.60,
        "held_out_split_ratio": 0.2,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "db_path": str(db_path),
        "random_seed": 42
    }
    return Config(config_dict)

@pytest.fixture
def test_db(mock_config):
    conn = sqlite3.connect(mock_config.db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE feature_versions (id INTEGER)")
    cursor.execute("CREATE TABLE items (id INTEGER)")
    cursor.execute("CREATE TABLE preference_log (id INTEGER)")
    cursor.execute("""
        CREATE TABLE model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            artifact_path TEXT NOT NULL,
            feature_version_id INTEGER NOT NULL,
            training_pair_count INTEGER NOT NULL,
            ndcg_at_10 REAL,
            map_score REAL,
            pairwise_accuracy REAL,
            is_bias_corrected INTEGER NOT NULL DEFAULT 0,
            trained_at TEXT NOT NULL DEFAULT (datetime('now')),
            is_active INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    return mock_config

def test_validate_environment_missing_db(mock_config, capsys):
    with pytest.raises(SystemExit):
        validate_environment(mock_config)
    captured = capsys.readouterr()
    assert "Database file not found" in captured.err

def test_validate_environment_missing_tables(mock_config, capsys):
    conn = sqlite3.connect(mock_config.db_path)
    conn.execute("CREATE TABLE items (id INTEGER)")
    conn.commit()
    conn.close()
    
    with pytest.raises(SystemExit):
        validate_environment(mock_config)
    captured = capsys.readouterr()
    assert "missing required tables" in captured.err

def test_rollback_invalid_id(test_db):
    with pytest.raises(ValueError, match="does not exist"):
        rollback_to_version(999, test_db)
        
def test_rollback_missing_file(test_db, tmp_path):
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, is_active)
        VALUES (?, 1, 10, 0)
    """, (str(tmp_path / "missing.pt"),))
    model_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    with pytest.raises(FileNotFoundError, match="artifact file is missing from disk"):
        rollback_to_version(model_id, test_db)

def test_rollback_success(test_db, tmp_path):
    # Create dummy artifact files
    art1 = tmp_path / "art1.pt"
    art1.write_text("dummy")
    art2 = tmp_path / "art2.pt"
    art2.write_text("dummy")
    
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, is_active)
        VALUES (?, 1, 10, 1)
    """, (str(art1),))
    
    cursor.execute("""
        INSERT INTO model_versions (artifact_path, feature_version_id, training_pair_count, is_active)
        VALUES (?, 1, 10, 0)
    """, (str(art2),))
    id2 = cursor.lastrowid
    conn.commit()
    conn.close()
    
    res = rollback_to_version(id2, test_db)
    assert res["model_version_id"] == id2
    
    # Verify DB state
    conn = sqlite3.connect(test_db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM model_versions WHERE is_active = 1")
    active_rows = cursor.fetchall()
    assert len(active_rows) == 1
    assert active_rows[0][0] == id2
    conn.close()

def test_cli_ingest_missing_csv():
    # Use python -m cli.main to run it
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    result = subprocess.run(
        ["python", "-m", "cli.main", "ingest"], 
        env=env,
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "the following arguments are required: --csv" in result.stderr

def test_cli_ingest_file_not_found():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    result = subprocess.run(
        ["python", "-m", "cli.main", "ingest", "--csv", "does_not_exist.csv"], 
        env=env,
        capture_output=True,
        text=True
    )
    assert result.returncode != 0
    assert "Error: CSV file not found at does_not_exist.csv" in result.stderr

def test_cli_rollback_no_yes(monkeypatch):
    # Should exit immediately if input is not 'yes' or 'y'
    # Actually, we can test this by running it in subprocess where stdin is closed.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    result = subprocess.run(
        ["python", "-m", "cli.main", "rollback", "--model-version-id", "1"], 
        env=env,
        input="n\n",
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Rollback cancelled." in result.stdout
