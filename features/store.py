import sqlite3
import json

def persist_item_features(df, feature_vectors, feature_version_id, config):
    """
    Writes item features to the database using UPSERT.
    """
    conn = sqlite3.connect(config.db_path)
    cursor = conn.cursor()
    
    records = []
    for i, row in df.iterrows():
        item_id = str(row['item_id'])
        text_desc = str(row.get('text_description', ''))
        
        vec_list = [float(x) for x in feature_vectors[i]]
        feature_vector_json = json.dumps(vec_list)
        
        records.append((
            item_id, 
            text_desc, 
            None, 
            None, 
            feature_vector_json, 
            feature_version_id
        ))
        
    cursor.execute("BEGIN TRANSACTION")
    try:
        cursor.executemany("""
            INSERT INTO items (
                item_id, text_description, numerical_features_json, 
                categorical_features_json, feature_vector_json, feature_version_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                text_description=excluded.text_description,
                numerical_features_json=excluded.numerical_features_json,
                categorical_features_json=excluded.categorical_features_json,
                feature_vector_json=excluded.feature_vector_json,
                feature_version_id=excluded.feature_version_id
        """, records)
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e
        
    conn.commit()
    conn.close()
