from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
from typing import Tuple


def create_basket_text(df: pd.DataFrame) -> pd.DataFrame:
    """Group items by Member_number and create a single basket text per member."""
    baskets = (
        df.groupby('Member_number')['itemDescription']
        .apply(lambda items: ' '.join(items.astype(str)))
        .reset_index(name='basket_text')
    )
    return baskets


def build_tfidf(texts, max_features: int = 500, min_df: int = 2, max_df: float = 0.8) -> Tuple[object, TfidfVectorizer]:
    """Build TF-IDF features with optimized parameters for better clustering."""
    vect = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        min_df=min_df,  # Ignore terms appearing in < 2 documents
        max_df=max_df,  # Ignore terms appearing in > 80% of documents
        ngram_range=(1, 2),  # Include unigrams and bigrams
        sublinear_tf=True  # Apply sublinear term frequency scaling
    )
    X = vect.fit_transform(texts)
    return X, vect


def apply_pca(X, n_components: int = 2):
    pca = PCA(n_components=n_components)
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    X_pca = pca.fit_transform(X_arr)
    return X_pca, pca


def apply_svd(X, n_components: int = 50):
    """Apply TruncatedSVD for dimensionality reduction (better for sparse data)."""
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_arr = X.toarray() if hasattr(X, 'toarray') else X
    X_svd = svd.fit_transform(X_arr)
    return X_svd, svd
