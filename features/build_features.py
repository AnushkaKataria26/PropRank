import numpy as np
import scipy.sparse
from features.text_features import fit_tfidf
from features.tabular_features import fit_numerical, fit_categorical

def build_feature_vectors(df, numerical_columns, categorical_columns, config):
    """
    Builds the final concatenated feature vectors for each item.
    Returns: feature_matrix (2D numpy array), vectorizer, scaler, encoder
    """
    max_features = config.tfidf_max_features
    
    # 1. Text features
    vectorizer, tfidf_sparse = fit_tfidf(df['text_description'], max_features=max_features)
    
    # 2. Numerical features
    scaler, num_array = fit_numerical(df, numerical_columns, fillna_strategy='raise')
    
    # 3. Categorical features
    encoder, cat_array = fit_categorical(df, categorical_columns)
    
    # Concatenate features: TF-IDF -> numerical -> categorical
    # Note: For very large catalogs, a sparse-preserving concatenation would be needed.
    # Out of scope for MVP, densifying TF-IDF block.
    if scipy.sparse.issparse(tfidf_sparse):
        tfidf_dense = tfidf_sparse.toarray()
    else:
        # It's our dummy vectorizer fallback or already dense
        if hasattr(tfidf_sparse, 'toarray'):
            tfidf_dense = tfidf_sparse.toarray()
        else:
            tfidf_dense = np.array(tfidf_sparse)
            
    blocks = []
    if tfidf_dense.shape[1] > 0:
        blocks.append(tfidf_dense)
    if num_array is not None and num_array.shape[1] > 0:
        blocks.append(num_array)
    if cat_array is not None and cat_array.shape[1] > 0:
        blocks.append(cat_array)
        
    if not blocks:
        feature_matrix = np.empty((len(df), 0))
    else:
        feature_matrix = np.hstack(blocks)
        
    return feature_matrix, vectorizer, scaler, encoder
