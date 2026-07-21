# PropRank

PropRank is a pairwise learning-to-rank system using RankNet with Inverse Propensity Weighting (IPW) for position bias correction and a BM25 fallback mechanism for cold-start scenarios. It handles feature engineering, pairwise preference simulation, model training, and dynamic inference in an end-to-end framework.

## Project Scope and Limitations
**Scope**: 
- Provides an automated pipeline for text-based feature extraction (TF-IDF).
- Simulates biased user click data.
- Trains a PyTorch RankNet MLP to predict pairwise item preferences.
- Uses IPW to unbias the click data based on position logs.
- Evaluates the model against baseline metrics (NDCG@10, Pairwise Accuracy, and MAP).
- Serves an inference engine with BM25 fallback logic when model confidence is low.
- Fully CLI-driven orchestration, retraining triggers, and automated validation.

**Limitations**:
- Does not scale to billion-scale datasets. Designed as a prototype/demo framework.
- Feature engineering is limited to TF-IDF vectorization and a hardcoded vocab size.
- Uses local SQLite for storage rather than a distributed feature store or cloud data warehouse.
- Relies on simulated preferences if real historical interaction data is not provided.

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or venv\Scripts\activate on Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Database**:
   ```bash
   python -m cli.main init-db
   ```
   *Note: This creates the SQLite database at the path specified in `config/config.json`.*

## CLI Command Reference
The project provides a fully integrated CLI (`cli/main.py`) with 8 subcommands:

- **`init-db`**: Initializes the SQLite schema.
- **`ingest`**: Ingests items from a CSV file into the database.
  - `--file`: Path to the CSV file.
- **`simulate`**: Generates synthetic preference logs (with configurable position bias).
  - `--contexts`: Number of contexts to simulate.
  - `--pairs`: Total number of pairs to generate.
- **`train`**: Trains a new model (Baseline or IPW-Corrected) on existing data.
  - `--bias-corrected`: Use IPW bias correction.
- **`retrain`**: Evaluates conditions (data volume) and conditionally retrains the model.
  - `--force`: Force retraining regardless of data volume triggers.
- **`inference`**: Runs the inference engine for a given query and context.
  - `--query`: Query text.
  - `--context`: Query context ID.
  - `--items`: Comma-separated list of candidate item IDs to rank.
- **`evaluate`**: Evaluates the currently active model and prints performance metrics.
- **`rollback`**: Deactivates the current active model and reverts to the previously active model.

*Example usage:*
```bash
python -m cli.main ingest --file demo/sample_data/demo_items.csv
python -m cli.main train --bias-corrected
```

## Running the End-to-End Demo

To run the complete demonstration encompassing Cold Start (Scenario 1), Warm Context (Scenario 2), and Retraining (Scenario 3):

```bash
python -m demo.run_full_demo
```
*This will ingest demo data, run inference scenarios showcasing the BM25 fallback and model activations, simulate new data, trigger a retraining cycle, evaluate the new model's performance, and benchmark inference latency.*

## Success Criteria Validation

PropRank aims to meet the following success criteria, which can be explicitly validated by running:
```bash
python -m demo.validate_success_criteria
```

**Success Metrics (Configurable via `config/config.json`)**:
- Post-training NDCG@10 >= 0.70
- BM25 Baseline NDCG@10 <= 0.55
- Pairwise Accuracy >= 0.70
- Inference latency for 1000 items <= 500ms
- Bias correction demonstrably reduces position-score correlation (visually verified in the comparison report).
