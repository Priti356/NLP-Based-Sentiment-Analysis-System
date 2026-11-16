"""
IntelliSentiment AI — NLP Preprocessing Pipeline
Cleans and normalises raw review text using NLTK and spaCy.
"""

import re
import logging
import string
from typing import List, Optional

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  NLTK Resource Download
# ------------------------------------------------------------------ #

def download_nltk_resources() -> None:
    """Download required NLTK corpora (idempotent)."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_resources()

try:
    import spacy
    try:
        NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed. Lemmatisation will use NLTK WordNetLemmatizer.")
    SPACY_AVAILABLE = False

from nltk.stem import WordNetLemmatizer
_lemmatizer = WordNetLemmatizer()

STOP_WORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------ #
#  Low-Level Text Cleaners
# ------------------------------------------------------------------ #

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_email_addresses(text: str) -> str:
    return re.sub(r"\S+@\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """Keep only letters, digits and spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'t": " not", "'ve": " have", "'m": " am",
        "it's": "it is", "i'm": "i am", "i've": "i have",
        "i'll": "i will", "i'd": "i would", "you're": "you are",
        "you've": "you have", "you'll": "you will", "you'd": "you would",
        "he's": "he is", "he'd": "he would", "he'll": "he will",
        "she's": "she is", "she'd": "she would", "she'll": "she will",
        "we're": "we are", "we've": "we have", "we'll": "we will",
        "we'd": "we would", "they're": "they are", "they've": "they have",
        "they'll": "they will", "they'd": "they would",
        "that's": "that is", "that'd": "that would", "that'll": "that will",
        "who's": "who is", "who'd": "who would", "who'll": "who will",
        "what's": "what is", "what're": "what are", "what'll": "what will",
        "where's": "where is", "when's": "when is", "why's": "why is",
        "how's": "how is",
    }
    for contraction, expansion in contractions.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------ #
#  Core Preprocessing Function
# ------------------------------------------------------------------ #

def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    min_token_length: int = 2,
) -> str:
    """
    Full NLP preprocessing pipeline for a single text string.

    Steps:
    1. Lowercase
    2. Expand contractions
    3. Remove URLs, HTML, emails
    4. Remove special characters / punctuation
    5. Tokenise
    6. Remove stopwords
    7. Lemmatise
    8. Rejoin tokens

    Returns:
        Cleaned, normalised text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Expand contractions
    text = expand_contractions(text)

    # Step 3: Remove noise
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_email_addresses(text)

    # Step 4: Remove special characters
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)

    # Step 5: Tokenise
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Step 6: Filter short tokens
    tokens = [t for t in tokens if len(t) >= min_token_length and t.isalpha()]

    # Step 7: Remove stopwords
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    # Step 8: Lemmatise
    if lemmatize:
        if SPACY_AVAILABLE:
            doc = NLP(" ".join(tokens))
            tokens = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS]
        else:
            tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------------ #
#  DataFrame-Level Preprocessing
# ------------------------------------------------------------------ #

def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "sentiment",
    cleaned_col: str = "cleaned_text",
    drop_duplicates: bool = True,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to a DataFrame.

    Args:
        df:               Input DataFrame.
        text_col:         Column containing raw text.
        label_col:        Column containing sentiment labels.
        cleaned_col:      Name of the new cleaned-text column.
        drop_duplicates:  Whether to remove duplicate texts.
        drop_missing:     Whether to drop rows with missing text/label.

    Returns:
        Preprocessed DataFrame with an additional ``cleaned_col`` column.
    """
    logger.info("Starting preprocessing. Input shape: %s", df.shape)

    # Drop missing values
    if drop_missing:
        before = len(df)
        df = df.dropna(subset=[text_col, label_col])
        logger.info("Dropped %d rows with missing values.", before - len(df))

    # Drop duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col])
        logger.info("Dropped %d duplicate rows.", before - len(df))

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Apply text preprocessing
    logger.info("Applying NLP preprocessing to %d reviews…", len(df))
    df[cleaned_col] = df[text_col].apply(preprocess_text)

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[df[cleaned_col].str.strip().str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d rows that became empty after cleaning.", before - len(df))

    logger.info("Preprocessing complete. Output shape: %s", df.shape)
    return df


def validate_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> dict:
    """
    Validate a reviews DataFrame and return a summary report.

    Returns:
        dict with keys: total, missing_text, missing_label, duplicates,
        label_distribution, issues.
    """
    issues = []
    report = {
        "total": len(df),
        "missing_text": int(df[text_col].isna().sum()) if text_col in df.columns else -1,
        "missing_label": int(df[label_col].isna().sum()) if label_col in df.columns else -1,
        "duplicates": int(df.duplicated(subset=[text_col]).sum()) if text_col in df.columns else -1,
        "label_distribution": {},
        "issues": [],
    }

    if text_col not in df.columns:
        issues.append(f"Missing required column: '{text_col}'")
    if label_col not in df.columns:
        issues.append(f"Missing required column: '{label_col}'")

    if not issues and label_col in df.columns:
        report["label_distribution"] = df[label_col].value_counts().to_dict()

    if report["missing_text"] > 0:
        issues.append(f"{report['missing_text']} rows have missing text.")
    if report["missing_label"] > 0:
        issues.append(f"{report['missing_label']} rows have missing labels.")
    if report["duplicates"] > 0:
        issues.append(f"{report['duplicates']} duplicate texts found.")

    report["issues"] = issues
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = "The product quality is excellent!! Delivery was VERY fast. I'll definitely buy again."
    print("Original :", sample)
    print("Cleaned  :", preprocess_text(sample))

