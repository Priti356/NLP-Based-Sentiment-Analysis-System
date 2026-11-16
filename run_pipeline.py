"""
IntelliSentiment AI — Full Training Pipeline Runner
Runs the complete pipeline end-to-end: data → preprocessing → EDA → training → evaluation.

Usage:
  python run_pipeline.py                      # run full pipeline with synthetic data
  python run_pipeline.py --data data/raw/reviews.csv  # use a real CSV
  python run_pipeline.py --samples 5000       # adjust synthetic dataset size
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def run_pipeline(data_path: str = None, n_samples: int = 3000):
    """
    Execute the full IntelliSentiment AI training pipeline.

    Steps:
    1. Initialise database
    2. Load dataset
    3. Validate dataset
    4. Preprocess text (NLP pipeline)
    5. Run EDA and save charts
    6. Train and compare 3 ML models
    7. Evaluate all models
    8. Save best model
    9. Persist evaluation metrics to DB
    """
    t_start = time.time()
    sep = "=" * 65

    print(f"\n{sep}")
    print("   IntelliSentiment AI — Full Pipeline")
    print(f"{sep}\n")

    # ---- Step 1: Database ----
    print("Step 1/9 — Initialising database…")
    from database.db import init_db, insert_model_evaluation, log_event
    db_path = str(BASE_DIR / "database" / "intellisentiment.db")
    init_db(db_path)
    log_event("pipeline", "Pipeline started.", db_path=db_path)
    print("  ✓ Database ready\n")

    # ---- Step 2: Load Data ----
    print(f"Step 2/9 — Loading dataset (n_samples={n_samples})…")
    from src.data_loader import load_dataset, save_processed_dataset
    df, source = load_dataset(file_path=data_path, n_synthetic=n_samples)
    print(f"  ✓ Loaded {len(df):,} reviews from: {source}")
    print(f"  ✓ Columns: {list(df.columns)}")
    print(f"  ✓ Label distribution: {df['sentiment'].value_counts().to_dict()}\n")

    # ---- Step 3: Validate ----
    print("Step 3/9 — Validating dataset…")
    from src.preprocessing import validate_dataset
    report = validate_dataset(df)
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ No data quality issues found")
    print()

    # ---- Step 4: Preprocess ----
    print("Step 4/9 — NLP Preprocessing pipeline…")
    from src.preprocessing import preprocess_dataframe
    t0 = time.time()
    df_clean = preprocess_dataframe(df)
    save_processed_dataset(df_clean)
    print(f"  ✓ Cleaned {len(df_clean):,} reviews in {time.time()-t0:.1f}s")
    print(f"  ✓ Sample cleaned text: '{df_clean['cleaned_text'].iloc[0][:70]}…'\n")

    # ---- Step 5: EDA ----
    print("Step 5/9 — Exploratory Data Analysis…")
    from src.eda import run_full_eda
    eda_results = run_full_eda(df_clean)
    stats = eda_results["stats"]
    print(f"  ✓ Total reviews   : {stats['total_reviews']:,}")
    print(f"  ✓ Positive        : {stats['positive_count']:,} ({stats['positive_pct']}%)")
    print(f"  ✓ Negative        : {stats['negative_count']:,} ({stats['negative_pct']}%)")
    print(f"  ✓ Neutral         : {stats['neutral_count']:,} ({stats['neutral_pct']}%)")
    print(f"  ✓ Avg word count  : {stats['avg_word_count']}")
    print(f"  ✓ Charts saved    : {len(eda_results['charts'])}\n")

    # ---- Step 6: Train ----
    print("Step 6/9 — Training ML models (LR, NB, SVM)…")
    from src.train import train_models
    t0 = time.time()
    training_results = train_models(df_clean)
    print(f"  ✓ Training complete in {time.time()-t0:.1f}s")
    for name, m in training_results["metrics"].items():
        marker = " ★" if name == training_results["best_model_name"] else "  "
        print(f" {marker} {name:35s} F1={m['macro_f1']:.4f} | Acc={m['accuracy']:.4f}")
    print()

    # ---- Step 7: Evaluate ----
    print("Step 7/9 — Generating evaluation reports & charts…")
    from src.evaluate import evaluate_all_models, print_evaluation_report
    eval_results = evaluate_all_models(training_results)
    print_evaluation_report(eval_results, training_results["best_model_name"])

    # ---- Step 8: Persist Metrics ----
    print("\nStep 8/9 — Saving evaluation metrics to database…")
    # Sort by macro_f1 descending so best model is first row
    sorted_models = sorted(eval_results.items(), key=lambda x: x[1]["macro_f1"], reverse=True)
    for name, m in sorted_models:
        insert_model_evaluation(
            model_name=name,
            accuracy=m["accuracy"],
            precision=m["macro_precision"],
            recall=m["macro_recall"],
            f1_score=m["macro_f1"],
            roc_auc=m.get("roc_auc"),
            notes=f"Best params: {training_results['metrics'][name].get('best_params', {})}",
            db_path=db_path,
        )
    print("  ✓ Metrics saved\n")

    # ---- Step 9: Summary ----
    total_time = time.time() - t_start
    best_name = training_results["best_model_name"]
    best_m    = eval_results[best_name]

    print(f"Step 9/9 — Pipeline complete!\n")
    print(f"{sep}")
    print(f"  BEST MODEL        : {best_name}")
    print(f"  Accuracy          : {best_m['accuracy']:.4f}")
    print(f"  Macro F1-Score    : {best_m['macro_f1']:.4f}")
    print(f"  Macro Precision   : {best_m['macro_precision']:.4f}")
    print(f"  Macro Recall      : {best_m['macro_recall']:.4f}")
    if best_m.get("roc_auc"):
        print(f"  ROC-AUC           : {best_m['roc_auc']:.4f}")
    print(f"  Total time        : {total_time:.1f}s")
    print(f"  Model saved to    : models/best_model.joblib")
    print(f"{sep}\n")

    log_event("pipeline", f"Pipeline completed. Best model: {best_name}", db_path=db_path)
    return training_results, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliSentiment AI Pipeline Runner")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--samples", type=int, default=3000, help="Number of synthetic samples (if no CSV)")
    args = parser.parse_args()

    try:
        run_pipeline(data_path=args.data, n_samples=args.samples)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)

