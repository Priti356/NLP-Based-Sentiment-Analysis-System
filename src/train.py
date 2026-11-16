"""
IntelliSentiment AI — Model Training Pipeline
Trains Logistic Regression, Multinomial Naive Bayes and SVM.
Selects the best model based on macro F1-score.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5


# ------------------------------------------------------------------ #
#  Pipeline Builders
# ------------------------------------------------------------------ #

def _build_pipelines(vectorizer) -> Dict[str, Pipeline]:
    """
    Return a dict of named Scikit-learn Pipelines, each containing
    the shared TF-IDF vectoriser step and a classifier.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("tfidf", vectorizer),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            )),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", vectorizer),
            ("clf", MultinomialNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("tfidf", vectorizer),
            ("clf", CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                ),
                cv=3,
            )),
        ]),
    }
    return pipelines


def _param_grids() -> Dict[str, Dict]:
    """Hyperparameter grids for GridSearchCV."""
    return {
        "Logistic Regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__solver": ["lbfgs"],
        },
        "Multinomial Naive Bayes": {
            "clf__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        "Support Vector Machine": {
            "clf__estimator__C": [0.1, 1.0, 5.0],
        },
    }


# ------------------------------------------------------------------ #
#  Main Training Function
# ------------------------------------------------------------------ #

def train_models(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    label_col: str = "sentiment",
    max_features: int = 20_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> dict:
    """
    Train all models on ``df`` and return a results dictionary.

    Args:
        df:            Preprocessed DataFrame with cleaned text and labels.
        text_col:      Column with preprocessed text.
        label_col:     Column with sentiment labels.
        max_features:  TF-IDF max vocabulary size.
        ngram_range:   TF-IDF n-gram range.

    Returns:
        dict with keys: best_model_name, best_model, models,
        metrics, X_test, y_test, label_classes
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Run preprocessing first.")

    X_raw = df[text_col].fillna("")
    y = df[label_col]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train size: %d | Test size: %d | Classes: %s",
        len(X_train), len(X_test), list(y.unique()),
    )

    # Build a fresh vectoriser for each pipeline (to avoid leakage)
    from sklearn.feature_extraction.text import TfidfVectorizer

    def _fresh_vec():
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
        )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    param_grids = _param_grids()

    all_metrics: Dict[str, dict] = {}
    trained_models: Dict[str, Pipeline] = {}

    for name in ["Logistic Regression", "Multinomial Naive Bayes", "Support Vector Machine"]:
        logger.info("Training: %s …", name)
        t0 = time.time()

        pipeline = _build_pipelines(_fresh_vec())[name]
        grid = param_grids.get(name, {})

        if grid:
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            cv_score = search.best_score_
            best_params = search.best_params_
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
            cv_score = None
            best_params = {}

        elapsed = time.time() - t0
        y_pred = best_pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Probabilities for ROC-AUC (if available)
        roc_auc = None
        try:
            from sklearn.metrics import roc_auc_score
            y_proba = best_pipeline.predict_proba(X_test)
            roc_auc = roc_auc_score(
                pd.get_dummies(y_test).values,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            pass

        all_metrics[name] = {
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc,
            "cv_f1": cv_score,
            "best_params": best_params,
            "train_time_s": round(elapsed, 2),
            "classification_report": report,
        }
        trained_models[name] = best_pipeline
        logger.info(
            "[%s] F1=%.4f | Accuracy=%.4f | Time=%.1fs",
            name, f1, report["accuracy"], elapsed,
        )

    # Select best model by macro F1-score
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["macro_f1"])
    best_model = trained_models[best_name]
    logger.info("Best model: %s (macro F1=%.4f)", best_name, all_metrics[best_name]["macro_f1"])

    # Persist best model and vectoriser
    best_model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    logger.info("Best model saved to %s", best_model_path)

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    # Save label classes
    label_classes = sorted(y.unique().tolist())
    joblib.dump(label_classes, MODEL_DIR / "label_classes.joblib")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "models": trained_models,
        "metrics": all_metrics,
        "X_test": X_test,
        "y_test": y_test,
        "label_classes": label_classes,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.data_loader import generate_synthetic_dataset
    from src.preprocessing import preprocess_dataframe

    df = generate_synthetic_dataset(1000)
    df = preprocess_dataframe(df)
    results = train_models(df)
    print("\n=== Training Complete ===")
    print(f"Best Model: {results['best_model_name']}")
    for name, m in results["metrics"].items():
        print(f"  {name}: F1={m['macro_f1']:.4f} | Accuracy={m['accuracy']:.4f}")

