"""
IntelliSentiment AI — Flask Application
Full-stack sentiment analysis web application.
"""

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# ------------------------------------
# Path setup so src/ and database/ are importable
# ------------------------------------
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.db import (
    get_sentiment_counts,
    get_sentiment_trend,
    get_recent_predictions,
    get_model_evaluations,
    init_db,
    insert_prediction,
    log_event,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("intellisentiment")

# ------------------------------------------------------------------ #
#  App Initialisation
# ------------------------------------------------------------------ #

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024    # 16 MB upload limit
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "intellisentiment-dev-key-2024")
CORS(app)

# Ensure the DB exists
init_db(str(BASE_DIR / "database" / "intellisentiment.db"))
DB_PATH = str(BASE_DIR / "database" / "intellisentiment.db")

# Lazy model loading
_model = None
_model_name = "Not Loaded"


def get_model():
    """Load the trained model on first request (lazy init)."""
    global _model, _model_name
    if _model is None:
        try:
            from src.predict import load_model
            _model = load_model()
            # Read best model name from evaluations db
            evals = get_model_evaluations(DB_PATH)
            if evals:
                _model_name = evals[0]["model_name"]
            else:
                _model_name = "Trained Model"
            logger.info("Model loaded: %s", _model_name)
        except FileNotFoundError:
            logger.warning("No trained model found. Run the training pipeline first.")
            _model = None
    return _model


# ------------------------------------------------------------------ #
#  Error Handlers
# ------------------------------------------------------------------ #

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", code=500, message="Internal server error."), 500


# ------------------------------------------------------------------ #
#  Home / Dashboard
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    """Main dashboard."""
    counts = get_sentiment_counts(DB_PATH)
    total  = sum(counts.values())
    pos    = counts.get("positive", 0)
    neg    = counts.get("negative", 0)
    neu    = counts.get("neutral", 0)

    pos_pct = round(pos / total * 100, 1) if total else 0
    neg_pct = round(neg / total * 100, 1) if total else 0
    neu_pct = round(neu / total * 100, 1) if total else 0

    recent = get_recent_predictions(10, DB_PATH)
    evals  = get_model_evaluations(DB_PATH)
    trend  = get_sentiment_trend(30, DB_PATH)

    model = get_model()
    model_ready = model is not None

    return render_template(
        "index.html",
        total=total,
        pos=pos, neg=neg, neu=neu,
        pos_pct=pos_pct, neg_pct=neg_pct, neu_pct=neu_pct,
        recent_predictions=recent,
        model_evaluations=evals,
        sentiment_trend=trend,
        model_ready=model_ready,
        model_name=_model_name,
    )


# ------------------------------------------------------------------ #
#  Single Prediction
# ------------------------------------------------------------------ #

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Single review prediction page and API."""
    if request.method == "GET":
        return render_template("predict.html")

    # JSON or form submission
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
    else:
        text = str(request.form.get("text", "")).strip()

    if not text:
        if request.is_json:
            return jsonify({"error": "No text provided."}), 400
        return render_template("predict.html", error="Please enter a review text.")

    model = get_model()
    if model is None:
        msg = "Model not trained. Run: python run_pipeline.py"
        if request.is_json:
            return jsonify({"error": msg}), 503
        return render_template("predict.html", error=msg)

    try:
        from src.predict import predict_single
        result = predict_single(text, model=model)

        # Persist to DB
        insert_prediction(
            raw_text=text,
            sentiment=result["sentiment"],
            cleaned_text=result.get("cleaned_text"),
            confidence=result.get("confidence"),
            model_used=_model_name,
            processing_ms=result.get("processing_ms"),
            db_path=DB_PATH,
        )
        log_event("prediction", f"Single prediction: {result['sentiment']}", db_path=DB_PATH)

        if request.is_json:
            return jsonify(result)
        return render_template("predict.html", result=result, input_text=text)

    except Exception as e:
        logger.exception("Prediction failed")
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template("predict.html", error=f"Prediction error: {e}")


# ------------------------------------------------------------------ #
#  Bulk CSV Prediction
# ------------------------------------------------------------------ #

@app.route("/bulk-predict", methods=["GET", "POST"])
def bulk_predict():
    """Bulk CSV sentiment prediction."""
    if request.method == "GET":
        return render_template("bulk_predict.html")

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted."}), 400

    model = get_model()
    if model is None:
        return jsonify({"error": "Model not trained. Run: python run_pipeline.py"}), 503

    try:
        content = file.read()
        text_col = request.form.get("text_col", "text")

        from src.predict import predict_from_csv
        df = predict_from_csv(content, text_col=text_col, model=model)

        # Store each prediction
        for _, row in df.iterrows():
            insert_prediction(
                raw_text=str(row.get(text_col, "")),
                sentiment=str(row["sentiment"]),
                cleaned_text=str(row.get("cleaned_text", "")),
                confidence=float(row["confidence"]) if row["confidence"] else None,
                model_used=_model_name,
                db_path=DB_PATH,
            )

        log_event("bulk_prediction", f"Processed {len(df)} rows from CSV.", db_path=DB_PATH)

        # Return summary and sample
        summary = df["sentiment"].value_counts().to_dict()
        sample  = df.head(20).to_dict(orient="records")
        return jsonify({
            "total_rows": len(df),
            "summary": summary,
            "sample": sample,
        })

    except Exception as e:
        logger.exception("Bulk prediction failed")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
#  Statistics / API
# ------------------------------------------------------------------ #

@app.route("/api/stats")
def api_stats():
    """Return current sentiment statistics as JSON."""
    counts = get_sentiment_counts(DB_PATH)
    total  = sum(counts.values())
    return jsonify({
        "total": total,
        "counts": counts,
        "percentages": {
            k: round(v / total * 100, 2) if total else 0
            for k, v in counts.items()
        },
    })


@app.route("/api/predictions")
def api_predictions():
    """Return recent predictions as JSON."""
    limit = min(int(request.args.get("limit", 50)), 500)
    preds = get_recent_predictions(limit, DB_PATH)
    return jsonify(preds)


@app.route("/api/trend")
def api_trend():
    """Return 30-day sentiment trend data."""
    days = int(request.args.get("days", 30))
    trend = get_sentiment_trend(days, DB_PATH)
    return jsonify(trend)


# ------------------------------------------------------------------ #
#  Model Evaluation Page
# ------------------------------------------------------------------ #

@app.route("/evaluation")
def evaluation():
    """Model evaluation results page."""
    evals = get_model_evaluations(DB_PATH)
    charts = {
        "model_comparison": "images/model_comparison.png",
        "confusion_matrix_lr": "images/confusion_matrix_logistic_regression.png",
        "confusion_matrix_nb": "images/confusion_matrix_multinomial_naive_bayes.png",
        "confusion_matrix_svm": "images/confusion_matrix_support_vector_machine.png",
    }
    # Only pass charts that actually exist
    existing_charts = {
        k: v for k, v in charts.items()
        if (BASE_DIR / "static" / v).exists()
    }
    return render_template("evaluation.html", evaluations=evals, charts=existing_charts)


# ------------------------------------------------------------------ #
#  Prediction History
# ------------------------------------------------------------------ #

@app.route("/history")
def history():
    """Prediction history page."""
    limit = min(int(request.args.get("limit", 100)), 1000)
    preds = get_recent_predictions(limit, DB_PATH)
    return render_template("history.html", predictions=preds, limit=limit)


# ------------------------------------------------------------------ #
#  Training Trigger (dev endpoint)
# ------------------------------------------------------------------ #

@app.route("/train", methods=["GET", "POST"])
def train():
    """Trigger model training via web interface."""
    if request.method == "GET":
        return render_template("train.html")

    logger.info("Training triggered via web interface.")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "run_pipeline.py")],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            global _model
            _model = None          # Force reload on next request
            log_event("training", "Training completed successfully.", db_path=DB_PATH)
            return jsonify({"status": "success", "output": result.stdout[-3000:]})
        else:
            log_event("training", f"Training failed: {result.stderr[:500]}", status="error", db_path=DB_PATH)
            return jsonify({"status": "error", "output": result.stderr[-3000:]}), 500
    except Exception as e:
        logger.exception("Training trigger failed")
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
#  EDA Charts Page
# ------------------------------------------------------------------ #

@app.route("/eda")
def eda():
    """EDA charts and statistics page."""
    chart_files = {
        "sentiment_distribution": "images/sentiment_distribution.png",
        "review_length": "images/review_length_distribution.png",
        "word_frequency": "images/word_frequency.png",
        "sentiment_trend": "images/sentiment_trend.png",
        "wordcloud_positive": "images/wordcloud_positive.png",
        "wordcloud_negative": "images/wordcloud_negative.png",
        "wordcloud_neutral": "images/wordcloud_neutral.png",
    }
    existing = {k: v for k, v in chart_files.items() if (BASE_DIR / "static" / v).exists()}
    return render_template("eda.html", charts=existing)


# ------------------------------------------------------------------ #
#  Entry Point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info("Starting IntelliSentiment AI on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)

