"""
IntelliSentiment AI — Test Suite
Tests for preprocessing, feature extraction, model prediction,
CSV validation, Flask API endpoints and database operations.
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ================================================================== #
#  Fixtures
# ================================================================== #

@pytest.fixture(scope="session")
def sample_df():
    """Small synthetic DataFrame for testing."""
    from src.data_loader import generate_synthetic_dataset
    return generate_synthetic_dataset(300, random_state=0)


@pytest.fixture(scope="session")
def cleaned_df(sample_df):
    """Preprocessed DataFrame."""
    from src.preprocessing import preprocess_dataframe
    return preprocess_dataframe(sample_df.copy())


@pytest.fixture(scope="session")
def trained_model(cleaned_df):
    """Train a quick model for prediction tests."""
    from src.train import train_models
    return train_models(cleaned_df, max_features=5000)


@pytest.fixture
def flask_client(tmp_path):
    """Flask test client with a temporary database."""
    from database.db import init_db
    db = str(tmp_path / "test.db")
    init_db(db)

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["SECRET_KEY"] = "test"
    # Override the global DB path used by the app
    flask_app.DB_PATH = db

    with flask_app.app.test_client() as client:
        yield client


# ================================================================== #
#  Preprocessing Tests
# ================================================================== #

class TestPreprocessing:

    def test_basic_cleaning(self):
        from src.preprocessing import preprocess_text
        raw = "The product is AMAZING!!! Visit http://example.com for more."
        result = preprocess_text(raw)
        assert isinstance(result, str)
        assert "http" not in result
        assert result == result.lower()

    def test_empty_string_returns_empty(self):
        from src.preprocessing import preprocess_text
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""

    def test_non_string_input(self):
        from src.preprocessing import preprocess_text
        assert preprocess_text(None) == ""

    def test_stopword_removal(self):
        from src.preprocessing import preprocess_text
        result = preprocess_text("the product is very good and amazing")
        assert "the" not in result.split()
        assert "is" not in result.split()
        assert "and" not in result.split()

    def test_url_removal(self):
        from src.preprocessing import preprocess_text, remove_urls
        text = "Check this out: https://www.example.com/product?id=123"
        cleaned = remove_urls(text)
        assert "https" not in cleaned
        assert "example.com" not in cleaned

    def test_special_chars_removal(self):
        from src.preprocessing import remove_special_characters
        text = "Hello, World! #Amazing @User $100"
        result = remove_special_characters(text)
        assert "#" not in result
        assert "@" not in result
        assert "$" not in result

    def test_dataframe_preprocessing(self, sample_df):
        from src.preprocessing import preprocess_dataframe
        df = preprocess_dataframe(sample_df.copy())
        assert "cleaned_text" in df.columns
        assert df["cleaned_text"].notna().all()
        assert len(df) > 0

    def test_validate_dataset_good(self, sample_df):
        from src.preprocessing import validate_dataset
        report = validate_dataset(sample_df)
        assert report["total"] == len(sample_df)
        assert isinstance(report["label_distribution"], dict)

    def test_validate_dataset_missing_col(self):
        from src.preprocessing import validate_dataset
        df = pd.DataFrame({"review": ["some text"], "label": ["positive"]})
        report = validate_dataset(df)  # defaults text_col='text'
        assert len(report["issues"]) > 0


# ================================================================== #
#  Feature Extraction Tests
# ================================================================== #

class TestFeatureExtraction:

    def test_tfidf_shape(self, cleaned_df):
        from src.feature_extraction import fit_tfidf
        vec, X = fit_tfidf(cleaned_df["cleaned_text"], max_features=1000, save=False)
        assert X.shape[0] == len(cleaned_df)
        assert X.shape[1] <= 1000

    def test_tfidf_no_save(self, cleaned_df):
        from src.feature_extraction import fit_tfidf
        vec, X = fit_tfidf(cleaned_df["cleaned_text"], max_features=500, save=False)
        assert len(vec.vocabulary_) > 0

    def test_text_statistics(self, cleaned_df):
        from src.feature_extraction import compute_text_statistics
        stats = compute_text_statistics(cleaned_df["cleaned_text"])
        assert "word_count" in stats.columns
        assert "char_count" in stats.columns
        assert stats["word_count"].min() >= 0


# ================================================================== #
#  Prediction Tests
# ================================================================== #

class TestPrediction:

    def test_single_positive(self, trained_model):
        from src.predict import predict_single
        model = trained_model["best_model"]
        result = predict_single(
            "This product is excellent and I love it!",
            model=model,
        )
        assert "sentiment" in result
        assert result["sentiment"] in ("positive", "negative", "neutral")
        assert isinstance(result.get("confidence"), (float, type(None)))

    def test_single_empty_text(self, trained_model):
        from src.predict import predict_single
        model = trained_model["best_model"]
        result = predict_single("", model=model)
        assert result["sentiment"] == "unknown"
        assert "error" in result

    def test_batch_prediction(self, trained_model):
        from src.predict import predict_batch
        model = trained_model["best_model"]
        texts = ["Great product!", "Terrible quality.", "It is okay."]
        results = predict_batch(texts, model=model)
        assert len(results) == 3
        for r in results:
            assert "sentiment" in r

    def test_csv_prediction(self, trained_model):
        from src.predict import predict_from_csv
        model = trained_model["best_model"]
        csv_bytes = b"text,rating\nGreat product!,5\nTerrible.,1\nOkay product.,3"
        df = predict_from_csv(csv_bytes, text_col="text", model=model)
        assert "sentiment" in df.columns
        assert len(df) == 3

    def test_csv_wrong_column(self, trained_model):
        from src.predict import predict_from_csv
        model = trained_model["best_model"]
        csv_bytes = b"review_text,rating\nGreat product!,5"
        # Should auto-detect 'review_text' column
        df = predict_from_csv(csv_bytes, text_col="text", model=model)
        assert "sentiment" in df.columns


# ================================================================== #
#  Database Tests
# ================================================================== #

class TestDatabase:

    def test_insert_and_retrieve_prediction(self, tmp_path):
        from database.db import init_db, insert_prediction, get_recent_predictions
        db = str(tmp_path / "test_db.db")
        init_db(db)
        insert_prediction("Great product", "positive", confidence=0.95, db_path=db)
        preds = get_recent_predictions(10, db_path=db)
        assert len(preds) == 1
        assert preds[0]["sentiment"] == "positive"

    def test_sentiment_counts(self, tmp_path):
        from database.db import init_db, insert_prediction, get_sentiment_counts
        db = str(tmp_path / "counts.db")
        init_db(db)
        insert_prediction("Review 1", "positive", db_path=db)
        insert_prediction("Review 2", "negative", db_path=db)
        insert_prediction("Review 3", "positive", db_path=db)
        counts = get_sentiment_counts(db_path=db)
        assert counts.get("positive") == 2
        assert counts.get("negative") == 1

    def test_model_evaluation_insert(self, tmp_path):
        from database.db import init_db, insert_model_evaluation, get_model_evaluations
        db = str(tmp_path / "eval.db")
        init_db(db)
        insert_model_evaluation(
            "Logistic Regression", 0.88, 0.87, 0.86, 0.87, 0.92,
            notes="test", db_path=db,
        )
        evals = get_model_evaluations(db_path=db)
        assert len(evals) == 1
        assert evals[0]["model_name"] == "Logistic Regression"

    def test_log_event(self, tmp_path):
        from database.db import init_db, log_event
        db = str(tmp_path / "log.db")
        init_db(db)
        # Should not raise
        log_event("test", "Test event", status="info", db_path=db)


# ================================================================== #
#  Flask API Endpoint Tests
# ================================================================== #

class TestFlaskAPI:

    def test_home_returns_200(self, flask_client):
        response = flask_client.get("/")
        assert response.status_code == 200

    def test_predict_page_get(self, flask_client):
        response = flask_client.get("/predict")
        assert response.status_code == 200

    def test_api_stats_json(self, flask_client):
        response = flask_client.get("/api/stats")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "total" in data
        assert "counts" in data

    def test_api_predictions_json(self, flask_client):
        response = flask_client.get("/api/predictions")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_predict_no_text(self, flask_client):
        response = flask_client.post(
            "/predict",
            json={"text": ""},
            content_type="application/json",
        )
        # Empty text — expect 400 or redirect with error
        assert response.status_code in (400, 200)

    def test_history_page(self, flask_client):
        response = flask_client.get("/history")
        assert response.status_code == 200

    def test_evaluation_page(self, flask_client):
        response = flask_client.get("/evaluation")
        assert response.status_code == 200

    def test_eda_page(self, flask_client):
        response = flask_client.get("/eda")
        assert response.status_code == 200

    def test_bulk_predict_no_file(self, flask_client):
        response = flask_client.post("/bulk-predict")
        assert response.status_code in (400, 200)

