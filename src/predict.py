"""
IntelliSentiment AI — Prediction Module
Single-review and batch CSV sentiment prediction.
"""

import csv
import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
LABEL_CLASSES_PATH = MODEL_DIR / "label_classes.joblib"


# ------------------------------------------------------------------ #
#  Model Loading
# ------------------------------------------------------------------ #

_model_cache: dict = {}   # module-level cache


def load_model(model_path: Optional[str] = None):
    """Load (and cache) the best trained pipeline from disk."""
    path = model_path or str(BEST_MODEL_PATH)
    if path not in _model_cache:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Run the training pipeline first: python src/train.py"
            )
        _model_cache[path] = joblib.load(path)
        logger.info("Model loaded from %s", path)
    return _model_cache[path]


def load_label_classes() -> List[str]:
    """Return the list of class labels used during training."""
    if LABEL_CLASSES_PATH.exists():
        return joblib.load(LABEL_CLASSES_PATH)
    return ["negative", "neutral", "positive"]     # sensible fallback


# ------------------------------------------------------------------ #
#  Single Review Prediction
# ------------------------------------------------------------------ #

def predict_single(
    text: str,
    model=None,
    preprocess: bool = True,
) -> Dict:
    """
    Predict the sentiment of a single review string.

    Args:
        text:       Raw review text.
        model:      Pre-loaded pipeline (loaded from disk if None).
        preprocess: Whether to run the NLP cleaning pipeline first.

    Returns:
        dict with keys: sentiment, confidence, probabilities, cleaned_text,
        processing_ms.
    """
    t0 = time.time()

    if not text or not text.strip():
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "cleaned_text": "",
            "processing_ms": 0.0,
            "error": "Empty input text.",
        }

    if model is None:
        model = load_model()

    cleaned = text
    if preprocess:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.preprocessing import preprocess_text
            cleaned = preprocess_text(text)
        except Exception as e:
            logger.warning("Preprocessing failed: %s. Using raw text.", e)

    prediction = model.predict([cleaned])[0]
    probabilities = {}
    confidence = None

    try:
        proba = model.predict_proba([cleaned])[0]
        label_classes = load_label_classes()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(label_classes, proba)}
        confidence = round(float(proba.max()), 4)
    except Exception:
        confidence = None

    elapsed_ms = round((time.time() - t0) * 1000, 2)

    return {
        "sentiment": str(prediction),
        "confidence": confidence,
        "probabilities": probabilities,
        "cleaned_text": cleaned,
        "processing_ms": elapsed_ms,
    }


# ------------------------------------------------------------------ #
#  Batch Prediction
# ------------------------------------------------------------------ #

def predict_batch(
    texts: List[str],
    model=None,
    preprocess: bool = True,
) -> List[Dict]:
    """
    Predict sentiment for a list of review strings.

    Returns:
        List of prediction dicts (same schema as predict_single).
    """
    if model is None:
        model = load_model()

    results = []
    for text in texts:
        result = predict_single(text, model=model, preprocess=preprocess)
        results.append(result)
    return results


# ------------------------------------------------------------------ #
#  CSV Bulk Prediction
# ------------------------------------------------------------------ #

def predict_from_csv(
    file_content: bytes,
    text_col: str = "text",
    model=None,
    preprocess: bool = True,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """
    Read a CSV file (bytes) and return a DataFrame with predictions.

    Args:
        file_content: Raw bytes of the uploaded CSV.
        text_col:     Column name containing review text.
        model:        Pre-loaded pipeline.
        max_rows:     Safety cap on number of rows to process.

    Returns:
        DataFrame with original columns + sentiment, confidence columns.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content), nrows=max_rows)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}") from e

    if text_col not in df.columns:
        # Fuzzy column match
        candidates = [c for c in df.columns if "text" in c.lower() or "review" in c.lower()]
        if candidates:
            text_col = candidates[0]
            logger.info("Using column '%s' as text column.", text_col)
        else:
            raise ValueError(
                f"Column '{text_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    if model is None:
        model = load_model()

    logger.info("Running batch prediction on %d rows…", len(df))
    results = predict_batch(df[text_col].fillna("").tolist(), model=model, preprocess=preprocess)

    df["sentiment"]  = [r["sentiment"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["cleaned_text"] = [r["cleaned_text"] for r in results]

    return df


# ------------------------------------------------------------------ #
#  Demo
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_reviews = [
        "The product quality is excellent and delivery was very fast.",
        "Terrible product. Broke after one day. Complete waste of money.",
        "It is okay. Nothing special but does the job.",
    ]

    model = load_model()
    for review in sample_reviews:
        result = predict_single(review, model=model)
        print(f"\nReview: {review[:60]}…")
        print(f"  Sentiment  : {result['sentiment'].upper()}")
        print(f"  Confidence : {result['confidence']:.1%}" if result["confidence"] else "  Confidence : N/A")
        print(f"  Probabilities: {result['probabilities']}")

