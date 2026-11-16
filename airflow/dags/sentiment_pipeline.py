"""
IntelliSentiment AI — Apache Airflow DAG
Scheduled daily sentiment analysis pipeline.

Workflow:
  1. load_data        — Load new review data
  2. validate_data    — Check dataset quality
  3. preprocess_data  — NLP text preprocessing
  4. run_predictions  — Generate batch predictions
  5. store_results    — Persist results to SQLite
  6. generate_report  — Build performance & sentiment summary

Requirements:
  pip install apache-airflow>=2.8.0

Setup:
  export AIRFLOW_HOME=./airflow
  airflow db init
  airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
  airflow webserver --port 8080  &
  airflow scheduler &

Then open http://localhost:8080 and enable the 'intellisentiment_pipeline' DAG.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)

# ------------------------------------
# Airflow Imports (guarded)
# ------------------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logger.warning("Airflow not installed. DAG definition skipped.")


# ------------------------------------------------------------------ #
#  Task Functions
# ------------------------------------------------------------------ #

def task_load_data(**context):
    """Task 1: Load review data from raw directory or generate synthetic."""
    logger.info("Task: load_data started")
    from src.data_loader import load_dataset, save_processed_dataset
    df, source = load_dataset()
    logger.info("Loaded %d rows from %s", len(df), source)

    # Save a dated snapshot for auditing
    dated_path = save_processed_dataset(df, f"raw_snapshot_{datetime.now().strftime('%Y%m%d')}.csv")
    context["ti"].xcom_push(key="n_rows", value=len(df))
    context["ti"].xcom_push(key="source", value=str(source))
    logger.info("Data loaded and snapshot saved: %s", dated_path)
    return {"n_rows": len(df), "source": str(source)}


def task_validate_data(**context):
    """Task 2: Validate the loaded dataset."""
    logger.info("Task: validate_data started")
    import pandas as pd
    from src.preprocessing import validate_dataset
    from src.data_loader import PROCESSED_DIR

    # Find latest snapshot
    files = sorted(PROCESSED_DIR.glob("raw_snapshot_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError("No raw snapshot found. Run load_data first.")

    df = pd.read_csv(files[0])
    report = validate_dataset(df)
    logger.info("Validation report: %s", report)

    if report.get("issues"):
        logger.warning("Dataset issues: %s", report["issues"])
    else:
        logger.info("Dataset is clean. No issues found.")

    context["ti"].xcom_push(key="validation_report", value=str(report))
    return report


def task_preprocess_data(**context):
    """Task 3: Run the full NLP preprocessing pipeline."""
    logger.info("Task: preprocess_data started")
    import pandas as pd
    from src.data_loader import PROCESSED_DIR, save_processed_dataset
    from src.preprocessing import preprocess_dataframe

    files = sorted(PROCESSED_DIR.glob("raw_snapshot_*.csv"), reverse=True)
    if not files:
        raise FileNotFoundError("No raw snapshot found.")

    df = pd.read_csv(files[0])
    df_clean = preprocess_dataframe(df)
    out_path = save_processed_dataset(df_clean, "processed_reviews.csv")

    logger.info("Preprocessing complete. Saved %d rows to %s", len(df_clean), out_path)
    context["ti"].xcom_push(key="n_clean", value=len(df_clean))
    return {"n_clean": len(df_clean)}


def task_run_predictions(**context):
    """Task 4: Generate batch predictions on the processed dataset."""
    logger.info("Task: run_predictions started")
    import pandas as pd
    from src.data_loader import load_processed_dataset
    from src.predict import load_model, predict_batch

    df = load_processed_dataset()
    model = load_model()

    texts = df["cleaned_text"].fillna("").tolist()
    results = predict_batch(texts, model=model)

    df["predicted_sentiment"] = [r["sentiment"] for r in results]
    df["confidence"]          = [r.get("confidence") for r in results]

    out_path = BASE_DIR / "data" / "processed" / f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out_path, index=False)

    summary = df["predicted_sentiment"].value_counts().to_dict()
    logger.info("Predictions generated: %s | saved to %s", summary, out_path)
    context["ti"].xcom_push(key="prediction_summary", value=str(summary))
    return {"prediction_summary": summary}


def task_store_results(**context):
    """Task 5: Persist batch predictions to SQLite."""
    logger.info("Task: store_results started")
    import pandas as pd
    from database.db import insert_prediction, log_event

    pred_files = sorted(
        (BASE_DIR / "data" / "processed").glob("predictions_*.csv"), reverse=True
    )
    if not pred_files:
        raise FileNotFoundError("No predictions file found.")

    df = pd.read_csv(pred_files[0])
    n_stored = 0
    for _, row in df.iterrows():
        insert_prediction(
            raw_text=str(row.get("text", "")),
            sentiment=str(row.get("predicted_sentiment", "unknown")),
            cleaned_text=str(row.get("cleaned_text", "")),
            confidence=float(row["confidence"]) if row.get("confidence") else None,
            model_used="airflow_pipeline",
            db_path=str(BASE_DIR / "database" / "intellisentiment.db"),
        )
        n_stored += 1

    log_event(
        "airflow_pipeline",
        f"Stored {n_stored} predictions from scheduled run.",
        db_path=str(BASE_DIR / "database" / "intellisentiment.db"),
    )
    logger.info("Stored %d predictions to database.", n_stored)
    return {"n_stored": n_stored}


def task_generate_report(**context):
    """Task 6: Generate a markdown sentiment summary report."""
    logger.info("Task: generate_report started")
    from database.db import get_sentiment_counts, get_model_evaluations

    db_path = str(BASE_DIR / "database" / "intellisentiment.db")
    counts  = get_sentiment_counts(db_path)
    total   = sum(counts.values())
    evals   = get_model_evaluations(db_path)

    report_lines = [
        f"# IntelliSentiment AI — Scheduled Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Sentiment Summary",
        f"| Sentiment | Count | % |",
        f"|-----------|-------|---|",
    ]
    for s, c in counts.items():
        pct = round(c / total * 100, 1) if total else 0
        report_lines.append(f"| {s.capitalize()} | {c:,} | {pct}% |")

    report_lines += ["", "## Best Model Performance"]
    if evals:
        best = evals[0]
        report_lines += [
            f"- **Model:** {best['model_name']}",
            f"- **Accuracy:** {best['accuracy']:.4f}",
            f"- **F1-Score:** {best['f1_score']:.4f}",
            f"- **Precision:** {best['precision']:.4f}",
            f"- **Recall:** {best['recall']:.4f}",
        ]

    report_path = BASE_DIR / "reports" / f"airflow_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    logger.info("Report generated: %s", report_path)
    return {"report_path": str(report_path)}


# ------------------------------------------------------------------ #
#  DAG Definition
# ------------------------------------------------------------------ #

if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "intellisentiment",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="intellisentiment_pipeline",
        default_args=default_args,
        description="Daily scheduled sentiment analysis pipeline",
        schedule_interval="0 2 * * *",    # 02:00 UTC every day
        start_date=days_ago(1),
        catchup=False,
        tags=["nlp", "sentiment", "ml"],
    ) as dag:

        t1 = PythonOperator(task_id="load_data",        python_callable=task_load_data)
        t2 = PythonOperator(task_id="validate_data",    python_callable=task_validate_data)
        t3 = PythonOperator(task_id="preprocess_data",  python_callable=task_preprocess_data)
        t4 = PythonOperator(task_id="run_predictions",  python_callable=task_run_predictions)
        t5 = PythonOperator(task_id="store_results",    python_callable=task_store_results)
        t6 = PythonOperator(task_id="generate_report",  python_callable=task_generate_report)

        t1 >> t2 >> t3 >> t4 >> t5 >> t6

