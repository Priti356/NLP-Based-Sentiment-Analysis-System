"""
IntelliSentiment AI — Dataset Loader and EDA Module
Loads, validates and explores the Amazon Product Reviews dataset.

Dataset: Amazon Product Reviews (sample)
Source  : https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
License : CC BY-NC-SA 4.0
Fallback: Generated synthetic dataset with realistic characteristics when
          the real dataset is unavailable.
"""

import logging
import random
import string
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ------------------------------------------------------------------ #
#  Synthetic Dataset Generator (fallback)
# ------------------------------------------------------------------ #

_POSITIVE_TEMPLATES = [
    "This product is absolutely amazing! Best purchase I have made in years.",
    "Excellent quality and fast delivery. Very satisfied with this item.",
    "Love this product! Works perfectly and looks great.",
    "Outstanding performance. Would definitely recommend to everyone.",
    "Great value for money. Exactly as described. Very happy customer!",
    "Perfect item. Arrived on time and in perfect condition.",
    "Exceeded my expectations. The quality is superb.",
    "Fantastic product! Will definitely buy again.",
    "Highly recommend this. Five stars all the way!",
    "Very happy with this purchase. Great customer service too.",
    "Best product in its category. Worth every penny.",
    "Quick delivery and product exactly as pictured. Very satisfied.",
    "Incredible quality at this price point. Highly satisfied.",
    "Works great and was easy to set up. Very impressed.",
    "Brilliant product, does exactly what it says. Very pleased.",
]

_NEGATIVE_TEMPLATES = [
    "Terrible product. Broke after one day of use. Complete waste of money.",
    "Very disappointed. The quality is nothing like the pictures showed.",
    "Do not buy this. It stopped working after a week.",
    "Awful experience. Product arrived damaged and customer service was unhelpful.",
    "Worst purchase I have ever made. Total rubbish.",
    "Poor quality materials. Fell apart immediately. Very disappointed.",
    "Not worth the money at all. Returned it right away.",
    "Complete garbage. Nothing like what was advertised.",
    "Extremely disappointed. Would give zero stars if possible.",
    "Horrible quality. Would not recommend to anyone.",
    "Broke almost immediately. Very poor construction.",
    "False advertising. Product is completely different from the description.",
    "Waste of time and money. Already stopped working.",
    "Very cheap feeling product. Definitely not as described.",
    "Disappointed with the quality. Would not purchase again.",
]

_NEUTRAL_TEMPLATES = [
    "It is okay. Nothing special but does the job.",
    "Average product. Gets the work done but nothing impressive.",
    "Decent quality for the price. Not the best but acceptable.",
    "It is fine. Works as expected, nothing more nothing less.",
    "Reasonable product. Has some issues but overall acceptable.",
    "Neither great nor terrible. Just an average product.",
    "Works as advertised. Not impressed but not disappointed either.",
    "It does what it is supposed to do. Nothing extraordinary.",
    "Average quality. Could be better but serves its purpose.",
    "Okay product. Met basic expectations but nothing exceptional.",
    "Fairly standard product. No major complaints, no major praise.",
    "Acceptable quality. Does the job well enough.",
    "Middle of the road product. Some good and some bad aspects.",
    "It is what it is. Nothing special but functional.",
    "Passable quality. Not the best option but acceptable.",
]


def generate_synthetic_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic but realistic sentiment dataset.

    Distribution: ~40% positive, ~35% negative, ~25% neutral
    (mimicking real-world review imbalance).
    """
    rng = random.Random(random_state)
    np.random.seed(random_state)

    positive_n = int(n_samples * 0.40)
    negative_n = int(n_samples * 0.35)
    neutral_n  = n_samples - positive_n - negative_n

    reviews, sentiments = [], []

    for template in rng.choices(_POSITIVE_TEMPLATES, k=positive_n):
        # Add mild variation
        variation = rng.choice(["", " Strongly recommend.", " Would buy again.", " 5 stars!", ""])
        reviews.append(template + variation)
        sentiments.append("positive")

    for template in rng.choices(_NEGATIVE_TEMPLATES, k=negative_n):
        variation = rng.choice(["", " Avoid this product.", " Total disappointment.", ""])
        reviews.append(template + variation)
        sentiments.append("negative")

    for template in rng.choices(_NEUTRAL_TEMPLATES, k=neutral_n):
        variation = rng.choice(["", " Take it or leave it.", " Could be improved.", ""])
        reviews.append(template + variation)
        sentiments.append("neutral")

    combined = list(zip(reviews, sentiments))
    rng.shuffle(combined)
    reviews, sentiments = zip(*combined)

    # Build DataFrame with metadata
    df = pd.DataFrame({
        "review_id": range(1, n_samples + 1),
        "text": reviews,
        "sentiment": sentiments,
        "rating": [
            5 if s == "positive" else (1 if s == "negative" else 3)
            for s in sentiments
        ],
        "product_category": rng.choices(
            ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports"],
            k=n_samples,
        ),
        "review_date": pd.date_range("2022-01-01", periods=n_samples, freq="2h").strftime("%Y-%m-%d").tolist(),
    })

    logger.info(
        "Generated synthetic dataset: %d reviews | Distribution: %s",
        len(df),
        df["sentiment"].value_counts().to_dict(),
    )
    return df


# ------------------------------------------------------------------ #
#  Dataset Loader
# ------------------------------------------------------------------ #

def load_dataset(
    file_path: Optional[str] = None,
    text_col: str = "text",
    label_col: str = "sentiment",
    n_synthetic: int = 3000,
) -> Tuple[pd.DataFrame, str]:
    """
    Load the reviews dataset from file or generate a synthetic one.

    Priority:
    1. ``file_path`` argument (CSV or TSV).
    2. Any CSV found in ``data/raw/``.
    3. Synthetic dataset (fallback).

    Returns:
        (DataFrame, source_description)
    """
    # 1) Explicit file path
    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, low_memory=False)
        logger.info("Loaded dataset from %s | Shape: %s", p, df.shape)
        return df, str(p)

    # 2) Scan raw data directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.tsv"))
    if csv_files:
        p = csv_files[0]
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, low_memory=False)
        logger.info("Auto-loaded dataset from %s | Shape: %s", p, df.shape)
        return df, str(p)

    # 3) Synthetic fallback
    logger.warning(
        "No CSV dataset found. Generating synthetic dataset with %d samples. "
        "To use real data, place a CSV file in data/raw/.",
        n_synthetic,
    )
    df = generate_synthetic_dataset(n_synthetic)
    return df, "synthetic"


def save_processed_dataset(df: pd.DataFrame, filename: str = "processed_reviews.csv") -> Path:
    """Save the preprocessed DataFrame to the processed data directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved processed dataset to %s | Shape: %s", out_path, df.shape)
    return out_path


def load_processed_dataset(filename: str = "processed_reviews.csv") -> pd.DataFrame:
    """Load the preprocessed dataset from disk."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run the preprocessing pipeline first."
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, source = load_dataset()
    print(f"\nDataset source: {source}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nSample rows:\n{df.head(3).to_string()}")

