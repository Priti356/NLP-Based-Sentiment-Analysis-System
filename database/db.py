"""
IntelliSentiment AI — Database Layer
Handles SQLite database creation, schema management and CRUD operations.
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "intellisentiment.db"


def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str = None) -> None:
    """Create all tables if they do not exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # ---- Reviews table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            text        TEXT    NOT NULL,
            source      TEXT    DEFAULT 'manual',
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ---- Sentiment Predictions table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id       INTEGER REFERENCES reviews(id) ON DELETE CASCADE,
            raw_text        TEXT    NOT NULL,
            cleaned_text    TEXT,
            sentiment       TEXT    NOT NULL,
            confidence      REAL,
            model_used      TEXT,
            processing_ms   REAL,
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ---- Model Evaluation Results table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT    NOT NULL,
            accuracy    REAL,
            precision   REAL,
            recall      REAL,
            f1_score    REAL,
            roc_auc     REAL,
            trained_at  TEXT    DEFAULT (datetime('now')),
            notes       TEXT
        )
    """)

    # ---- Processing Logs table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT    NOT NULL,
            message     TEXT,
            status      TEXT    DEFAULT 'info',
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


# ------------------------------------------------------------------ #
#  Reviews
# ------------------------------------------------------------------ #

def insert_review(text: str, source: str = "manual", db_path: str = None) -> int:
    """Insert a raw review and return its id."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO reviews (text, source) VALUES (?, ?)",
        (text, source)
    )
    conn.commit()
    review_id = cur.lastrowid
    conn.close()
    return review_id


def get_all_reviews(limit: int = 500, db_path: str = None) -> List[Dict]:
    """Fetch the most recent reviews."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM reviews ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
#  Predictions
# ------------------------------------------------------------------ #

def insert_prediction(
    raw_text: str,
    sentiment: str,
    cleaned_text: str = None,
    confidence: float = None,
    model_used: str = None,
    processing_ms: float = None,
    review_id: int = None,
    db_path: str = None,
) -> int:
    """Store a single prediction and return its id."""
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO predictions
           (review_id, raw_text, cleaned_text, sentiment, confidence, model_used, processing_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (review_id, raw_text, cleaned_text, sentiment, confidence, model_used, processing_ms),
    )
    conn.commit()
    pred_id = cur.lastrowid
    conn.close()
    return pred_id


def get_recent_predictions(limit: int = 50, db_path: str = None) -> List[Dict]:
    """Return the most recent predictions."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sentiment_counts(db_path: str = None) -> Dict[str, int]:
    """Return counts per sentiment label."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT sentiment, COUNT(*) as cnt FROM predictions GROUP BY sentiment"
    ).fetchall()
    conn.close()
    return {r["sentiment"]: r["cnt"] for r in rows}


def get_sentiment_trend(days: int = 30, db_path: str = None) -> List[Dict]:
    """Return daily sentiment counts for the last N days."""
    conn = get_connection(db_path)
    rows = conn.execute(
        """SELECT DATE(created_at) as day, sentiment, COUNT(*) as cnt
           FROM predictions
           WHERE created_at >= datetime('now', ?)
           GROUP BY day, sentiment
           ORDER BY day""",
        (f"-{days} days",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
#  Model Evaluations
# ------------------------------------------------------------------ #

def insert_model_evaluation(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    roc_auc: float = None,
    notes: str = None,
    db_path: str = None,
) -> None:
    """Persist evaluation metrics for a trained model."""
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO model_evaluations
           (model_name, accuracy, precision, recall, f1_score, roc_auc, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (model_name, accuracy, precision, recall, f1_score, roc_auc, notes),
    )
    conn.commit()
    conn.close()


def get_model_evaluations(db_path: str = None) -> List[Dict]:
    """Fetch all stored model evaluation records."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM model_evaluations ORDER BY trained_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #

def log_event(event_type: str, message: str, status: str = "info", db_path: str = None) -> None:
    """Write an application event to the processing_logs table."""
    try:
        conn = get_connection(db_path)
        conn.execute(
            "INSERT INTO processing_logs (event_type, message, status) VALUES (?, ?, ?)",
            (event_type, message, status),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Could not write log event: %s", exc)


# ------------------------------------------------------------------ #
#  Utility SQL Queries
# ------------------------------------------------------------------ #

SENTIMENT_SUMMARY_SQL = """
SELECT
    sentiment,
    COUNT(*)                                    AS total,
    ROUND(AVG(confidence) * 100, 2)             AS avg_confidence_pct,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM predictions
GROUP BY sentiment
ORDER BY total DESC;
"""

DAILY_TREND_SQL = """
SELECT
    DATE(created_at) AS day,
    sentiment,
    COUNT(*)         AS total
FROM predictions
GROUP BY day, sentiment
ORDER BY day DESC
LIMIT 90;
"""

TOP_MODELS_SQL = """
SELECT model_name, accuracy, precision, recall, f1_score, roc_auc, trained_at
FROM model_evaluations
ORDER BY f1_score DESC;
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialised successfully.")
    print(f"DB location: {DB_PATH}")

