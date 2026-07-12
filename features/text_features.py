import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

logger = logging.getLogger(__name__)

class DummyVectorizer:
    """Fallback vectorizer when text corpus is empty."""
    def transform(self, raw_documents):
        import scipy.sparse
        return scipy.sparse.csr_matrix((len(raw_documents), 0))

def fit_tfidf(text_series, max_features=500):
    """
    Fits TF-IDF on text descriptions.
    """
    text_series = text_series.fillna("").astype(str)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(text_series)
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            logger.warning("Text signal is empty or corpus is too small to build vocabulary. Falling back to zero-width text features.")
            n_items = len(text_series)
            tfidf_matrix = scipy.sparse.csr_matrix((n_items, 0))
            vectorizer = DummyVectorizer()
        else:
            raise e
            
    return vectorizer, tfidf_matrix
