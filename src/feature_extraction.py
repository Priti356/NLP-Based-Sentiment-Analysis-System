"""
IntelliSentiment AI — Feature Extraction
TF-IDF vectorisation with Scikit-learn Pipeline utilities.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import spmatrix

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"


# ------------------------------------------------------------------ #
#  TF-IDF Vectoriser
# ------------------------------------------------------------------ #

def build_tfidf_vectorizer(
    max_features: int = 20_000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Instantiate a TF-IDF vectoriser with sensible defaults.

    Args:
        max_features:  Maximum vocabulary size.
        ngram_range:   Unigrams + bigrams by default.
        min_df:        Minimum document frequency.
        max_df:        Maximum document frequency (removes overly common terms).
        sublinear_tf:  Apply log(1 + tf) scaling.

    Returns:
        Unfitted TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
    )


def fit_tfidf(
    texts: pd.Series,
    max_features: int = 20_000,
    ngram_range: Tuple[int, int] = (1, 2),
    save: bool = True,
) -> Tuple[TfidfVectorizer, spmatrix]:
    """
    Fit a TF-IDF vectoriser on a text Series and return
    both the fitted vectoriser and the feature matrix.

    Args:
        texts:         Cleaned text Series.
        max_features:  Vocabulary size cap.
        ngram_range:   N-gram range tuple.
        save:          Whether to persist the vectoriser to disk.

    Returns:
        (vectorizer, X_tfidf)
    """
    vectorizer = build_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    X = vectorizer.fit_transform(texts.fillna(""))
    logger.info(
        "TF-IDF fitted. Vocabulary size: %d | Matrix shape: %s",
        len(vectorizer.vocabulary_),
        X.shape,
    )

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, TFIDF_PATH)
        logger.info("TF-IDF vectoriser saved to %s", TFIDF_PATH)

    return vectorizer, X


def transform_tfidf(
    texts: pd.Series,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> spmatrix:
    """
    Transform text using an already-fitted TF-IDF vectoriser.

    If no vectoriser is provided, the saved one from disk is loaded.
    """
    if vectorizer is None:
        if not TFIDF_PATH.exists():
            raise FileNotFoundError(
                f"No fitted TF-IDF vectoriser found at {TFIDF_PATH}. "
                "Train the model first."
            )
        vectorizer = joblib.load(TFIDF_PATH)
        logger.info("Loaded TF-IDF vectoriser from %s", TFIDF_PATH)

    return vectorizer.transform(texts.fillna(""))


def load_tfidf_vectorizer() -> TfidfVectorizer:
    """Load and return the persisted TF-IDF vectoriser."""
    if not TFIDF_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectoriser not found at {TFIDF_PATH}")
    return joblib.load(TFIDF_PATH)


# ------------------------------------------------------------------ #
#  Label Encoding
# ------------------------------------------------------------------ #

def encode_labels(labels: pd.Series, save: bool = True) -> Tuple[LabelEncoder, np.ndarray]:
    """
    Fit-encode sentiment labels (e.g. positive → 2, negative → 0, neutral → 1).

    Returns:
        (encoder, encoded_array)
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    logger.info("Label classes: %s", list(encoder.classes_))

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, LABEL_ENCODER_PATH)
        logger.info("Label encoder saved to %s", LABEL_ENCODER_PATH)

    return encoder, y


def load_label_encoder() -> LabelEncoder:
    """Load the persisted LabelEncoder from disk."""
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
    return joblib.load(LABEL_ENCODER_PATH)


# ------------------------------------------------------------------ #
#  Feature Engineering Helpers
# ------------------------------------------------------------------ #

def get_top_features(
    vectorizer: TfidfVectorizer,
    n: int = 30,
) -> pd.DataFrame:
    """
    Return the top N features (terms) by average TF-IDF weight.

    Useful for understanding what vocabulary the model learned.
    """
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame({"term": feature_names[:n], "index": range(n)})


def compute_text_statistics(texts: pd.Series) -> pd.DataFrame:
    """
    Compute basic text statistics useful for EDA.

    Returns a DataFrame with columns:
    - char_count
    - word_count
    - avg_word_length
    - unique_word_count
    """
    stats = pd.DataFrame()
    stats["char_count"] = texts.str.len()
    stats["word_count"] = texts.str.split().str.len()
    stats["avg_word_length"] = texts.apply(
        lambda t: np.mean([len(w) for w in str(t).split()]) if pd.notna(t) and str(t).split() else 0
    )
    stats["unique_word_count"] = texts.apply(
        lambda t: len(set(str(t).split())) if pd.notna(t) else 0
    )
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_texts = pd.Series([
        "product excellent fast delivery",
        "terrible quality waste money",
        "average product nothing special",
    ])
    vec, X = fit_tfidf(sample_texts, max_features=100, save=False)
    print("Feature matrix shape:", X.shape)
    print("Sample features:", vec.get_feature_names_out()[:10])

