import os
import random
import csv
import sqlite3
from config.loader import get_config
from features.run_pipeline import run_feature_pipeline

def setup_demo_data(num_items=500):
    config = get_config()
    random.seed(config.random_seed)

    demo_dir = "demo/sample_data"
    os.makedirs(demo_dir, exist_ok=True)
    csv_path = os.path.join(demo_dir, "demo_items.csv")

    # Generate synthetic demo data if not exists
    if not os.path.exists(csv_path):
        print(f"Generating synthetic demo data at {csv_path}...")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['item_id', 'text_description', 'price', 'category'])
            categories = ['electronics', 'books', 'clothing', 'home']
            for i in range(1, num_items + 1):
                item_id = f"item_{i}"
                text_description = f"This is a description for item {i} which is a great {random.choice(categories)} product."
                price = round(random.uniform(10.0, 500.0), 2)
                category = random.choice(categories)
                writer.writerow([item_id, text_description, price, category])
        print("Data generation complete.")
    else:
        print(f"Using existing demo data at {csv_path}.")

    # Check if this data is already in the database
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    
    # ensure db schema exists in case this is the first run
    from db.init_db import init_db
    init_db()

    # Check if we have items with these IDs already
    cursor.execute("SELECT COUNT(*), MAX(feature_version_id) FROM items WHERE item_id LIKE 'item_%'")
    count, max_fv_id = cursor.fetchone()
    conn.close()

    if count > 0 and max_fv_id is not None:
        print(f"Found {count} existing demo items in database with feature_version_id={max_fv_id}. Reusing.")
        feature_version_id = max_fv_id
    else:
        print("Running feature pipeline to ingest demo data...")
        feature_version_id = run_feature_pipeline(
            csv_path=csv_path,
            numerical_columns=['price'],
            categorical_columns=['category']
        )
    
    contexts = {
        'cold': 'ctx_cold',
        'warm': 'ctx_warm_gradual',
        'retrain': 'ctx_retrain'
    }

    print(f"Setup complete. Active feature_version_id: {feature_version_id}")
    return feature_version_id, contexts

if __name__ == "__main__":
    setup_demo_data()
