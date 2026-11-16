"""
IntelliSentiment AI — Kafka Consumer
Reads review messages from Kafka, predicts sentiment using the trained
NLP model, and stores results in the SQLite database.

Architecture:
  Kafka Topic (sentiment-reviews) → Consumer → NLP Model → SQLite DB

Usage:
  python kafka/consumer.py          # run consumer (Ctrl+C to stop)
  python kafka/consumer.py --dry-run # process without storing to DB
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC_REVIEWS     = "sentiment-reviews"
KAFKA_GROUP_ID          = "sentiment-consumer-group"
DB_PATH                 = str(BASE_DIR / "database" / "intellisentiment.db")

# Global flag for graceful shutdown
_running = True


def signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received. Stopping consumer…")
    _running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer(topic: str = KAFKA_TOPIC_REVIEWS):
    """Create and return a KafkaConsumer."""
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,         # 1 s poll timeout
            max_poll_records=50,
        )
        logger.info("KafkaConsumer connected. Subscribed to '%s'", topic)
        return consumer
    except Exception as e:
        logger.error("Could not connect to Kafka: %s", e)
        logger.error("Ensure Kafka is running. See README for Docker setup.")
        return None


def load_model_once():
    """Load the trained model (once at startup)."""
    try:
        from src.predict import load_model
        model = load_model()
        logger.info("Sentiment model loaded successfully.")
        return model
    except FileNotFoundError:
        logger.error("Trained model not found. Run the training pipeline first.")
        return None


def process_message(raw_message: dict, model, dry_run: bool = False) -> dict:
    """
    Process a single Kafka message:
    1. Extract review text.
    2. Run NLP prediction.
    3. Store result in DB (unless dry_run).

    Returns the prediction result dict.
    """
    text = str(raw_message.get("text", "")).strip()
    if not text:
        return {"error": "Empty message text.", "message_id": raw_message.get("message_id")}

    t0 = time.time()
    try:
        from src.predict import predict_single
        result = predict_single(text, model=model)
        result["message_id"] = raw_message.get("message_id")
        result["source"]     = raw_message.get("source", "kafka")
        result["kafka_ts"]   = raw_message.get("timestamp")

        if not dry_run:
            from database.db import insert_prediction, log_event
            insert_prediction(
                raw_text=text,
                sentiment=result["sentiment"],
                cleaned_text=result.get("cleaned_text"),
                confidence=result.get("confidence"),
                model_used="best_model",
                processing_ms=result.get("processing_ms"),
                db_path=DB_PATH,
            )

        elapsed = round((time.time() - t0) * 1000, 2)
        logger.info(
            "[Consumer] msg_id=%s | sentiment=%s | confidence=%.3f | %.1f ms",
            result["message_id"], result["sentiment"],
            result.get("confidence") or 0.0, elapsed,
        )
        return result

    except Exception as e:
        logger.error("Error processing message %s: %s", raw_message.get("message_id"), e)
        return {"error": str(e), "message_id": raw_message.get("message_id")}


def run_consumer(dry_run: bool = False, max_messages: int = None):
    """
    Main consumer loop.

    Args:
        dry_run:      If True, predict but do not write to DB.
        max_messages: Stop after this many messages (None = run indefinitely).
    """
    consumer = create_consumer()
    if consumer is None:
        return

    model = load_model_once()

    logger.info("Consumer started. Listening for reviews… (Ctrl+C to stop)")
    processed = 0
    errors    = 0

    try:
        while _running:
            if max_messages and processed >= max_messages:
                logger.info("Reached max_messages limit (%d). Stopping.", max_messages)
                break

            for message in consumer:
                if not _running:
                    break
                raw = message.value
                result = process_message(raw, model, dry_run=dry_run)
                if "error" in result:
                    errors += 1
                else:
                    processed += 1

    except Exception as e:
        logger.error("Consumer error: %s", e)
    finally:
        consumer.close()
        logger.info(
            "Consumer stopped. Processed: %d | Errors: %d",
            processed, errors,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliSentiment Kafka Consumer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Predict without writing to database")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of messages to process")
    args = parser.parse_args()

    run_consumer(dry_run=args.dry_run, max_messages=args.max)

