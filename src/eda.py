"""
IntelliSentiment AI — EDA Module
Exploratory Data Analysis: statistics, charts, word clouds.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

STATIC_IMAGES = Path(__file__).parent.parent / "static" / "images"
STATIC_IMAGES.mkdir(parents=True, exist_ok=True)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#3498db",
}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ------------------------------------------------------------------ #
#  Summary Statistics
# ------------------------------------------------------------------ #

def compute_summary_stats(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> dict:
    """Return a dictionary of key EDA statistics."""
    counts = df[label_col].value_counts().to_dict()
    total = len(df)

    stats = {
        "total_reviews": total,
        "positive_count": counts.get("positive", 0),
        "negative_count": counts.get("negative", 0),
        "neutral_count": counts.get("neutral", 0),
        "positive_pct": round(counts.get("positive", 0) / total * 100, 1),
        "negative_pct": round(counts.get("negative", 0) / total * 100, 1),
        "neutral_pct": round(counts.get("neutral", 0) / total * 100, 1),
        "avg_review_length": round(df[text_col].str.len().mean(), 1),
        "median_review_length": round(df[text_col].str.len().median(), 1),
        "avg_word_count": round(df[text_col].str.split().str.len().mean(), 1),
        "unique_reviews": df[text_col].nunique(),
    }
    return stats


# ------------------------------------------------------------------ #
#  Plot Helpers
# ------------------------------------------------------------------ #

def _save_fig(fig: plt.Figure, filename: str) -> Path:
    out = STATIC_IMAGES / filename
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved chart: %s", out)
    return out


def plot_sentiment_distribution(df: pd.DataFrame, label_col: str = "sentiment") -> Path:
    """Bar chart of sentiment class counts."""
    counts = df[label_col].value_counts()
    colors = [SENTIMENT_COLORS.get(s, "#95a5a6") for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}\n({val/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Reviews", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.2)
    return _save_fig(fig, "sentiment_distribution.png")


def plot_review_length_distribution(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> Path:
    """Histogram of review character lengths, separated by sentiment."""
    df = df.copy()
    df["char_length"] = df[text_col].str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    for sentiment, grp in df.groupby(label_col):
        ax.hist(
            grp["char_length"].clip(upper=1000),
            bins=40,
            alpha=0.6,
            label=sentiment,
            color=SENTIMENT_COLORS.get(sentiment, "#95a5a6"),
        )

    ax.set_title("Review Length Distribution by Sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Character Count", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Sentiment")
    return _save_fig(fig, "review_length_distribution.png")


def plot_word_frequency(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
    top_n: int = 20,
) -> Path:
    """Horizontal bar chart of top-N most frequent words per sentiment."""
    if text_col not in df.columns:
        text_col = "text"

    sentiments = df[label_col].unique()
    ncols = len(sentiments)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    for ax, sentiment in zip(axes, sorted(sentiments)):
        texts = " ".join(df[df[label_col] == sentiment][text_col].dropna())
        words = [w for w in texts.split() if len(w) > 2]
        counter = Counter(words).most_common(top_n)
        if not counter:
            continue
        terms, freqs = zip(*counter)
        color = SENTIMENT_COLORS.get(sentiment, "#95a5a6")
        ax.barh(terms[::-1], freqs[::-1], color=color, edgecolor="white")
        ax.set_title(f"Top {top_n} Words — {sentiment.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequency")

    fig.suptitle("Most Frequent Words by Sentiment", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "word_frequency.png")


def plot_sentiment_trend(df: pd.DataFrame, date_col: str = "review_date", label_col: str = "sentiment") -> Optional[Path]:
    """Line chart of sentiment counts over time."""
    if date_col not in df.columns:
        logger.warning("No '%s' column; skipping trend plot.", date_col)
        return None

    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception:
        return None

    trend = (
        df.groupby([pd.Grouper(key=date_col, freq="W"), label_col])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for sentiment, grp in trend.groupby(label_col):
        ax.plot(
            grp[date_col], grp["count"],
            label=sentiment,
            color=SENTIMENT_COLORS.get(sentiment, "#95a5a6"),
            linewidth=2, marker="o", markersize=4,
        )

    ax.set_title("Weekly Sentiment Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Reviews")
    ax.legend(title="Sentiment")
    fig.autofmt_xdate()
    return _save_fig(fig, "sentiment_trend.png")


def plot_wordcloud(
    df: pd.DataFrame,
    sentiment: str = "positive",
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
) -> Optional[Path]:
    """Generate a word cloud for the given sentiment."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud package not installed; skipping word cloud.")
        return None

    if text_col not in df.columns:
        text_col = "text"

    subset = df[df[label_col] == sentiment][text_col].dropna()
    text = " ".join(subset)
    if not text.strip():
        return None

    color = SENTIMENT_COLORS.get(sentiment, "#7f8c8d").lstrip("#")
    r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn" if sentiment == "positive" else ("Reds" if sentiment == "negative" else "Blues"),
        max_words=100,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {sentiment.capitalize()} Reviews", fontsize=14, fontweight="bold")
    filename = f"wordcloud_{sentiment}.png"
    return _save_fig(fig, filename)


def run_full_eda(df: pd.DataFrame, text_col: str = "text", label_col: str = "sentiment") -> dict:
    """
    Run all EDA steps and return a results dictionary with
    statistics and paths to generated charts.
    """
    logger.info("Running full EDA on %d records…", len(df))
    results = {
        "stats": compute_summary_stats(df, text_col, label_col),
        "charts": {},
    }

    cleaned_col = "cleaned_text" if "cleaned_text" in df.columns else text_col

    try:
        results["charts"]["sentiment_distribution"] = str(
            plot_sentiment_distribution(df, label_col)
        )
    except Exception as e:
        logger.error("sentiment_distribution failed: %s", e)

    try:
        results["charts"]["review_length"] = str(
            plot_review_length_distribution(df, text_col, label_col)
        )
    except Exception as e:
        logger.error("review_length failed: %s", e)

    try:
        results["charts"]["word_frequency"] = str(
            plot_word_frequency(df, cleaned_col, label_col)
        )
    except Exception as e:
        logger.error("word_frequency failed: %s", e)

    try:
        path = plot_sentiment_trend(df, label_col=label_col)
        if path:
            results["charts"]["sentiment_trend"] = str(path)
    except Exception as e:
        logger.error("sentiment_trend failed: %s", e)

    for sentiment in df[label_col].unique():
        try:
            path = plot_wordcloud(df, sentiment, cleaned_col, label_col)
            if path:
                results["charts"][f"wordcloud_{sentiment}"] = str(path)
        except Exception as e:
            logger.error("wordcloud_%s failed: %s", sentiment, e)

    logger.info("EDA complete. Charts: %s", list(results["charts"].keys()))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import generate_synthetic_dataset
    df = generate_synthetic_dataset(500)
    results = run_full_eda(df)
    print("Stats:", results["stats"])
    print("Charts:", list(results["charts"].keys()))

