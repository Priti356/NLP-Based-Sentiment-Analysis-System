"""
IntelliSentiment AI — Kafka Producer
Publishes customer review messages to a Kafka topic in real time.

Architecture:
  Review Producer → Kafka Topic (sentiment-reviews) → Consumer → NLP → DB

Setup (Docker Compose):
  See project README for docker-compose.yml to start Kafka + Zookeeper.

Usage:
  python kafka/producer.py --mode demo         # send 20 sample reviews
  python kafka/producer.py --mode interactive  # enter reviews manually
  python kafka/producer.py --mode csv --file data/raw/reviews.csv
"""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_REVIEWS     = "sentiment-reviews"

SAMPLE_REVIEWS = [
    ("The product is absolutely amazing! Best purchase I have made in years.", "positive"),
    ("Terrible quality. Broke after one day. Complete waste of money.",        "negative"),
    ("It is okay. Nothing special but does the job.",                          "neutral"),
    ("Excellent quality and fast delivery. Very satisfied!",                   "positive"),
    ("Do not buy this. Customer service was unhelpful and rude.",              "negative"),
    ("Average product. Gets the work done but nothing impressive.",            "neutral"),
    ("Love this product! Works perfectly and looks great.",                    "positive"),
    ("Very disappointed. The quality is nothing like the pictures.",           "negative"),
    ("Decent quality for the price. Not the best but acceptable.",             "neutral"),
    ("Outstanding performance. Would definitely recommend to everyone.",       "positive"),
]


def build_message(text: str, source: str = "producer", category: str = "general") -> dict:
    """Build a JSON-serialisable review message."""
    return {
        "message_id": str(uuid.uuid4()),
        "text": text,
        "source": source,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_producer():
    """Create and return a KafkaProducer instance."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
            request_timeout_ms=5000,
        )
        logger.info("KafkaProducer connected to %s", KAFKA_BOOTSTRAP_SERVERS)
        return producer
    except Exception as e:
        logger.error("Could not connect to Kafka: %s", e)
        logger.error("Ensure Kafka is running. See README for Docker setup.")
        return None


def send_review(producer, text: str, topic: str = KAFKA_TOPIC_REVIEWS) -> bool:
    """Send a single review message to the Kafka topic."""
    if producer is None:
        logger.error("Producer is not connected.")
        return False
    try:
        msg = build_message(text)
        future = producer.send(topic, value=msg)
        producer.flush(timeout=5)
        meta = future.get(timeout=5)
        logger.info(
            "Sent → topic=%s | partition=%d | offset=%d | msg_id=%s",
            meta.topic, meta.partition, meta.offset, msg["message_id"],
        )
        return True
    except Exception as e:
        logger.error("Failed to send message: %s", e)
        return False


def run_demo_mode(producer, delay: float = 0.5):
    """Send all sample reviews as a demo batch."""
    logger.info("Running demo mode: sending %d sample reviews…", len(SAMPLE_REVIEWS))
    sent = 0
    for text, label in SAMPLE_REVIEWS:
        if send_review(producer, text):
            sent += 1
        time.sleep(delay)
    logger.info("Demo complete: %d/%d messages sent.", sent, len(SAMPLE_REVIEWS))
    return sent


def run_interactive_mode(producer):
    """Allow user to type reviews interactively."""
    print("\nInteractive Mode — type a review and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            text = input("Review: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            break
        send_review(producer, text)


def run_csv_mode(producer, file_path: str, text_col: str = "text", delay: float = 0.1):
    """Stream reviews from a CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    if text_col not in df.columns:
        candidates = [c for c in df.columns if "text" in c.lower() or "review" in c.lower()]
        if candidates:
            text_col = candidates[0]
        else:
            logger.error("Column '%s' not found in %s", text_col, file_path)
            return
    logger.info("Streaming %d reviews from %s…", len(df), file_path)
    for _, row in df.iterrows():
        send_review(producer, str(row[text_col]))
        time.sleep(delay)
    logger.info("CSV streaming complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliSentiment Kafka Producer")
    parser.add_argument("--mode", choices=["demo", "interactive", "csv"], default="demo")
    parser.add_argument("--file", help="CSV file path (for csv mode)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between messages (seconds)")
    args = parser.parse_args()

    producer = create_producer()
    if producer is None:
        sys.exit(1)

    try:
        if args.mode == "demo":
            run_demo_mode(producer, delay=args.delay)
        elif args.mode == "interactive":
            run_interactive_mode(producer)
        elif args.mode == "csv":
            if not args.file:
                logger.error("--file is required for csv mode.")
                sys.exit(1)
            run_csv_mode(producer, args.file, delay=args.delay)
    finally:
        if producer:
            producer.close()
            logger.info("Producer closed.")

