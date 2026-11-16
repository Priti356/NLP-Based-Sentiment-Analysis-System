"""
IntelliSentiment AI — PySpark Big Data Processing Module

Demonstrates scalable sentiment data processing using Apache Spark.
Compare processing speed between Pandas (single-node) and
PySpark (distributed-ready, running on local[*] for demo purposes).

Requirements:
  pip install pyspark>=3.5.0

Usage:
  python spark/spark_processing.py
"""

import logging
import time
from pathlib import Path
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


# ------------------------------------------------------------------ #
#  PySpark Availability Check
# ------------------------------------------------------------------ #

def check_spark_available() -> bool:
    try:
        import pyspark
        return True
    except ImportError:
        logger.error("PySpark not installed. Run: pip install pyspark>=3.5.0")
        return False


# ------------------------------------------------------------------ #
#  Spark Session Builder
# ------------------------------------------------------------------ #

def create_spark_session(app_name: str = "IntelliSentimentSpark", master: str = "local[*]"):
    """Create and return a SparkSession."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("SparkSession created: %s (Spark %s)", app_name, spark.version)
    return spark


# ------------------------------------------------------------------ #
#  Data Loading
# ------------------------------------------------------------------ #

def load_data_spark(spark, csv_path: str = None, n_synthetic: int = 5000):
    """
    Load review data into a Spark DataFrame.

    Uses a CSV file if provided, otherwise generates synthetic data
    and loads it via Pandas → Spark conversion.
    """
    from pyspark.sql import functions as F

    if csv_path and Path(csv_path).exists():
        logger.info("Loading CSV into Spark: %s", csv_path)
        sdf = spark.read.csv(csv_path, header=True, inferSchema=True)
    else:
        logger.info("Generating %d synthetic records for Spark demo…", n_synthetic)
        from src.data_loader import generate_synthetic_dataset
        import pandas as pd
        pdf = generate_synthetic_dataset(n_synthetic)
        sdf = spark.createDataFrame(pdf)

    logger.info("Spark DataFrame schema:")
    sdf.printSchema()
    logger.info("Row count: %d", sdf.count())
    return sdf


# ------------------------------------------------------------------ #
#  Distributed Text Cleaning
# ------------------------------------------------------------------ #

def clean_text_spark(sdf, text_col: str = "text"):
    """
    Apply text cleaning transformations using Spark SQL functions
    (lowercase, punctuation removal, whitespace normalisation).
    """
    from pyspark.sql import functions as F

    sdf_clean = (
        sdf
        .filter(F.col(text_col).isNotNull())
        .filter(F.length(F.trim(F.col(text_col))) > 0)
        .withColumn("cleaned_spark",
            F.trim(
                F.regexp_replace(
                    F.lower(F.col(text_col)),
                    r"[^a-zA-Z0-9\s]", " "
                )
            )
        )
        .withColumn("word_count", F.size(F.split(F.col("cleaned_spark"), r"\s+")))
        .withColumn("char_count", F.length(F.col(text_col)))
        .dropDuplicates([text_col])
    )
    logger.info("Text cleaning applied. Sample:")
    sdf_clean.select(text_col, "cleaned_spark", "word_count").show(5, truncate=60)
    return sdf_clean


# ------------------------------------------------------------------ #
#  Sentiment Statistics (Distributed)
# ------------------------------------------------------------------ #

def compute_sentiment_stats_spark(sdf, label_col: str = "sentiment"):
    """
    Compute sentiment distribution and word-count statistics
    grouped by sentiment label using Spark aggregations.
    """
    from pyspark.sql import functions as F

    stats = (
        sdf.groupBy(label_col)
        .agg(
            F.count("*").alias("count"),
            F.avg("word_count").alias("avg_word_count"),
            F.avg("char_count").alias("avg_char_count"),
            F.min("word_count").alias("min_word_count"),
            F.max("word_count").alias("max_word_count"),
        )
        .orderBy(F.col("count").desc())
    )

    total = sdf.count()
    stats = stats.withColumn("percentage", F.round(F.col("count") / total * 100, 2))
    logger.info("Sentiment statistics:")
    stats.show()
    return stats


# ------------------------------------------------------------------ #
#  Pandas vs Spark Comparison Benchmark
# ------------------------------------------------------------------ #

def benchmark_pandas_vs_spark(n_rows: int = 10_000) -> dict:
    """
    Compare processing time for the same task (cleaning + groupby)
    between Pandas and PySpark on a local dataset of n_rows reviews.

    Returns:
        dict with pandas_seconds, spark_seconds, speedup_ratio
    """
    import pandas as pd
    import re
    from src.data_loader import generate_synthetic_dataset

    df_pd = generate_synthetic_dataset(n_rows)

    # ---- Pandas benchmark ----
    logger.info("Benchmarking Pandas on %d rows…", n_rows)
    t0 = time.time()
    df_pd["cleaned"] = df_pd["text"].str.lower().str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True).str.strip()
    df_pd["word_count"] = df_pd["cleaned"].str.split().str.len()
    _ = df_pd.groupby("sentiment").agg({"word_count": ["mean", "count"]})
    pandas_time = time.time() - t0
    logger.info("Pandas time: %.3f s", pandas_time)

    # ---- PySpark benchmark ----
    logger.info("Benchmarking PySpark on %d rows…", n_rows)
    if not check_spark_available():
        return {"pandas_seconds": pandas_time, "spark_seconds": None, "speedup_ratio": None}

    spark = create_spark_session()
    t0 = time.time()
    sdf = spark.createDataFrame(df_pd)
    sdf_clean = clean_text_spark(sdf)
    compute_sentiment_stats_spark(sdf_clean)
    sdf_clean.count()          # trigger actual execution
    spark_time = time.time() - t0
    logger.info("PySpark time: %.3f s", spark_time)

    spark.stop()

    return {
        "n_rows": n_rows,
        "pandas_seconds": round(pandas_time, 3),
        "spark_seconds": round(spark_time, 3),
        "note": (
            "Spark has JVM startup overhead on small datasets. "
            "Spark excels at datasets with millions of rows and cluster deployments."
        ),
    }


# ------------------------------------------------------------------ #
#  Full Spark Pipeline
# ------------------------------------------------------------------ #

def run_spark_pipeline(csv_path: str = None, n_synthetic: int = 5000) -> dict:
    """
    End-to-end Spark sentiment data processing pipeline.

    Returns a summary dict with statistics.
    """
    if not check_spark_available():
        return {"error": "PySpark not available."}

    spark = create_spark_session()
    try:
        sdf = load_data_spark(spark, csv_path, n_synthetic)
        sdf_clean = clean_text_spark(sdf)
        stats_sdf = compute_sentiment_stats_spark(sdf_clean, "sentiment")

        stats_dict = {}
        for row in stats_sdf.collect():
            stats_dict[row["sentiment"]] = {
                "count": row["count"],
                "avg_word_count": round(row["avg_word_count"], 2),
                "avg_char_count": round(row["avg_char_count"], 2),
                "percentage": row["percentage"],
            }

        total_rows = sdf_clean.count()
        logger.info("Spark pipeline complete. Total rows processed: %d", total_rows)
        return {"total_rows": total_rows, "sentiment_stats": stats_dict}

    finally:
        spark.stop()
        logger.info("SparkSession stopped.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  IntelliSentiment AI — PySpark Processing Demo")
    print("=" * 60)

    result = run_spark_pipeline(n_synthetic=3000)
    if "error" in result:
        print(f"\nError: {result['error']}")
        print("Install PySpark: pip install pyspark>=3.5.0")
    else:
        print(f"\nTotal rows processed: {result['total_rows']:,}")
        print("\nSentiment Statistics:")
        for sentiment, stats in result["sentiment_stats"].items():
            print(f"  {sentiment:10s} — count: {stats['count']:,} ({stats['percentage']}%) | avg_words: {stats['avg_word_count']}")

    print("\n" + "-" * 60)
    print("Running Pandas vs PySpark benchmark (5000 rows)…")
    bench = benchmark_pandas_vs_spark(5000)
    print(f"  Pandas  : {bench['pandas_seconds']} s")
    print(f"  PySpark : {bench.get('spark_seconds', 'N/A')} s")
    print(f"  Note    : {bench.get('note','')}")

