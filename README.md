# PropRank

PropRank is a pairwise learning-to-rank system using RankNet with position bias correction and BM25 fallback. It handles feature engineering, pairwise preference simulation, and model training in an end-to-end framework.

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
   python db/init_db.py
   ```

## Phases

- Phase 0 - Project Scaffolding
- Phase 1 - Database & Feature Registry
- Phase 2 - Preferences Simulation
- Phase 3 - Training Pipeline
- Phase 4 - Model Artifacts & Evaluation
- Phase 5 - BM25 Fallback
- Phase 6 - Inference Engine
- Phase 7 - CLI Integration
