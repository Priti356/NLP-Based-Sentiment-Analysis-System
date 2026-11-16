# 🧠 IntelliSentiment AI
### Advanced NLP-Based Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-blue)](https://spacy.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Project Overview

**IntelliSentiment AI** is a production-ready, end-to-end sentiment analysis platform that classifies customer reviews, product feedback, and social media text into **Positive**, **Negative**, and **Neutral** categories using state-of-the-art NLP and Machine Learning techniques.

The platform demonstrates real-world skills in:
- Natural Language Processing (NLP)
- Machine Learning model development and evaluation
- Big Data processing with Apache Spark
- Real-time streaming with Apache Kafka
- Workflow orchestration with Apache Airflow
- Full-stack Python web development with Flask

---

## 🎯 Business Problem

Companies receive thousands of customer reviews daily. Manual analysis is impossible at scale.
**IntelliSentiment AI** automates sentiment classification, enabling:
- Real-time customer feedback monitoring
- Automated escalation for negative reviews
- Trend analysis for product improvement
- Competitive intelligence gathering

---

## 🏗️ Architecture

```
Customer Reviews (CSV / API / Kafka)
         │
         ▼
┌─────────────────────────────────────┐
│   NLP Preprocessing Pipeline        │
│   (Clean → Tokenise → Lemmatise)    │
└──────────────┬──────────────────────┘
               │
         TF-IDF Vectorisation
               │
    ┌──────────▼──────────┐
    │   ML Model Training  │
    │  ┌────────────────┐  │
    │  │ Logistic Reg.  │  │
    │  │ Naive Bayes    │  │
    │  │ SVM (Linear)   │  │
    │  └────────────────┘  │
    │  GridSearchCV + CV   │
    └──────────┬───────────┘
               │
        Best Model (F1)
               │
    ┌──────────▼───────────┐    ┌────────────┐
    │   Flask Backend       │◄──►│  SQLite DB  │
    │   REST API            │    └────────────┘
    └──────────┬────────────┘
               │
    ┌──────────▼────────────┐
    │  Interactive Dashboard │
    │  (Bootstrap + Chart.js)│
    └───────────────────────┘

Optional Extensions:
  Apache Spark  →  Distributed data processing
  Apache Kafka  →  Real-time review streaming
  Apache Airflow → Scheduled pipeline execution
```

---

## ⚙️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **NLP** | NLTK, spaCy, TF-IDF, Tokenisation, Lemmatisation |
| **ML** | Scikit-learn, Logistic Regression, Naive Bayes, SVM |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix |
| **Big Data** | Apache Spark, PySpark |
| **Streaming** | Apache Kafka, kafka-python |
| **Orchestration** | Apache Airflow |
| **Backend** | Flask, Flask-CORS, SQLite, SQLAlchemy |
| **Visualization** | Matplotlib, Seaborn, Plotly, Chart.js, WordCloud |
| **Frontend** | Bootstrap 5, HTML5, CSS3, JavaScript |
| **Testing** | pytest, pytest-flask |

---

## 📁 Project Structure

```
IntelliSentiment-AI/
│
├── app.py                      ← Flask web application (main entry point)
├── run_pipeline.py             ← Full ML training pipeline runner
├── requirements.txt            ← Python dependencies
├── pytest.ini                  ← Test configuration
├── README.md
├── .gitignore
├── .env.example                ← Environment variable template
│
├── data/
│   ├── raw/                    ← Place your CSV dataset here
│   └── processed/              ← Cleaned datasets and predictions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          ← Dataset loading + synthetic generation
│   ├── preprocessing.py        ← Full NLP preprocessing pipeline
│   ├── feature_extraction.py   ← TF-IDF vectorisation + feature engineering
│   ├── eda.py                  ← Exploratory Data Analysis + charts
│   ├── train.py                ← ML model training + hyperparameter tuning
│   ├── evaluate.py             ← Model evaluation + confusion matrix charts
│   └── predict.py              ← Single and batch prediction interface
│
├── models/
│   ├── best_model.joblib       ← Best trained model (auto-generated)
│   ├── logistic_regression.joblib
│   ├── multinomial_naive_bayes.joblib
│   ├── support_vector_machine.joblib
│   └── label_classes.joblib
│
├── database/
│   ├── __init__.py
│   ├── db.py                   ← SQLite schema, CRUD operations
│   └── intellisentiment.db     ← SQLite database (auto-created)
│
├── templates/                  ← Jinja2 HTML templates
│   ├── base.html
│   ├── index.html              ← Dashboard
│   ├── predict.html            ← Single prediction page
│   ├── bulk_predict.html       ← CSV upload page
│   ├── evaluation.html         ← Model evaluation report
│   ├── eda.html                ← EDA charts page
│   ├── history.html            ← Prediction history
│   ├── train.html              ← Training trigger page
│   └── error.html
│
├── static/
│   ├── css/style.css           ← Global stylesheet
│   ├── js/main.js              ← Global JavaScript
│   └── images/                 ← Generated charts (auto-populated)
│
├── spark/
│   └── spark_processing.py     ← PySpark distributed processing module
│
├── kafka/
│   ├── producer.py             ← Kafka review message producer
│   └── consumer.py             ← Kafka consumer + NLP prediction
│
├── airflow/
│   └── dags/
│       └── sentiment_pipeline.py ← Airflow DAG (daily scheduled pipeline)
│
└── tests/
    ├── __init__.py
    └── test_intellisentiment.py  ← Full test suite
```

---

## 🚀 Installation (Windows)

### Prerequisites
- Python 3.9+ ([python.org](https://www.python.org/downloads/))
- pip (bundled with Python)
- Git (optional)

### Step 1 — Clone or navigate to the project
```powershell
cd c:\path\to\IntelliSentiment-AI
```

### Step 2 — Create a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies
```powershell
pip install -r requirements.txt
```

### Step 4 — Download spaCy language model
```powershell
python -m spacy download en_core_web_sm
```

### Step 5 — Copy environment variables
```powershell
copy .env.example .env
```

---

## 📊 Dataset

### Default: Synthetic Dataset (Auto-generated)
If no CSV is provided, the system auto-generates a **3,000-review synthetic dataset** with realistic characteristics and class imbalance (~40% positive, ~35% negative, ~25% neutral).

### Using a Real Dataset

#### Option A — Amazon Product Reviews (Kaggle)
1. Download from: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
2. Place the CSV file in `data/raw/`
3. The system auto-detects and loads it

#### Option B — Your Own Dataset
Your CSV must have at least:
- A `text` column (review text)
- A `sentiment` column (`positive`, `negative`, `neutral`)

```powershell
python run_pipeline.py --data data\raw\your_reviews.csv
```

#### Option C — NLTK Datasets
```python
import nltk
nltk.download('movie_reviews')  # 2000 movie reviews
```

---

## 🏃 Running the Project

### Full Pipeline (Training + EDA + Evaluation)
```powershell
python run_pipeline.py
```

Output:
```
=================================================================
   IntelliSentiment AI — Full Pipeline
=================================================================

Step 1/9 — Initialising database…
  ✓ Database ready

Step 2/9 — Loading dataset (n_samples=3000)…
  ✓ Loaded 3,000 reviews from: synthetic
  ✓ Label distribution: {'positive': 1200, 'negative': 1050, 'neutral': 750}

Step 3/9 — Validating dataset…
  ✓ No data quality issues found

Step 4/9 — NLP Preprocessing pipeline…
  ✓ Cleaned 2,998 reviews in 12.4s

Step 5/9 — Exploratory Data Analysis…
  ✓ Total reviews: 2,998
  ✓ Positive: 1,200 (40.0%)
  ✓ Negative: 1,049 (35.0%)
  ✓ Neutral:  749 (25.0%)

Step 6/9 — Training ML models (LR, NB, SVM)…
  ★ Logistic Regression          F1=0.8923 | Acc=0.8950
    Multinomial Naive Bayes      F1=0.8615 | Acc=0.8700
    Support Vector Machine       F1=0.8847 | Acc=0.8900

Step 7/9 — Generating evaluation reports…
  [Evaluation report printed to console]

Step 8/9 — Saving metrics to database…
Step 9/9 — Pipeline complete!

BEST MODEL: Logistic Regression
Macro F1  : 0.8923
Accuracy  : 0.8950
```

> **Note:** Actual results depend on dataset. Numbers above are illustrative.
> Run the pipeline to see real model performance.

### Launch Flask Web Application
```powershell
python app.py
```
Then open http://localhost:5000 in your browser.

### Run Tests
```powershell
pytest tests/ -v
```

---

## 🌐 API Documentation

### `GET /`
Dashboard with KPI cards, charts and recent predictions.

### `POST /predict`
Single review prediction.

**Request (JSON):**
```json
{ "text": "The product quality is excellent and delivery was fast." }
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.9412,
  "probabilities": {
    "negative": 0.0215,
    "neutral": 0.0373,
    "positive": 0.9412
  },
  "cleaned_text": "product quality excellent delivery fast",
  "processing_ms": 12.4
}
```

### `POST /bulk-predict`
CSV file upload for batch predictions.

**Form fields:**
- `file`: CSV file (required)
- `text_col`: Column name (default: `text`)

**Response:**
```json
{
  "total_rows": 150,
  "summary": { "positive": 90, "negative": 35, "neutral": 25 },
  "sample": [...]
}
```

### `GET /api/stats`
Current sentiment statistics.

### `GET /api/predictions?limit=50`
Recent predictions (JSON).

### `GET /api/trend?days=30`
Daily sentiment trend data.

---

## 🔥 Apache Spark Module

```powershell
python spark\spark_processing.py
```

Features:
- Distributed text cleaning with SparkSQL functions
- Sentiment statistics via Spark aggregations
- Pandas vs PySpark benchmark comparison
- Runs on `local[*]` mode (no cluster required)

---

## 📡 Apache Kafka Module

### Prerequisites: Start Kafka (Docker)

Create `docker-compose.yml` in the project root:
```yaml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

```powershell
docker-compose up -d
```

### Run Producer (send sample reviews):
```powershell
python kafka\producer.py --mode demo
```

### Run Consumer (predict and store):
```powershell
python kafka\consumer.py
```

---

## 🌊 Apache Airflow Module

### Installation
```powershell
pip install apache-airflow>=2.8.0
```

### Setup
```powershell
$env:AIRFLOW_HOME = "$pwd\airflow"
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080
# In a new terminal:
airflow scheduler
```

Open http://localhost:8080, enable `intellisentiment_pipeline` DAG.

### DAG Schedule
Runs daily at 02:00 UTC: load → validate → preprocess → predict → store → report.

---

## 🔒 Database Schema

```sql
-- Reviews
CREATE TABLE reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  source TEXT DEFAULT 'manual',
  created_at TEXT DEFAULT (datetime('now'))
);

-- Predictions
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  review_id INTEGER REFERENCES reviews(id),
  raw_text TEXT NOT NULL,
  cleaned_text TEXT,
  sentiment TEXT NOT NULL,
  confidence REAL,
  model_used TEXT,
  processing_ms REAL,
  created_at TEXT DEFAULT (datetime('now'))
);

-- Model Evaluations
CREATE TABLE model_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_name TEXT NOT NULL,
  accuracy REAL, precision REAL, recall REAL,
  f1_score REAL, roc_auc REAL,
  trained_at TEXT DEFAULT (datetime('now')),
  notes TEXT
);
```

---

## 📈 Model Evaluation

### Metric Selection Rationale

| Metric | Why Used |
|--------|----------|
| **Macro F1** | Primary metric — treats all classes equally, crucial for imbalanced data |
| **Accuracy** | Secondary — can be misleading with imbalance |
| **Precision** | Measures false positive rate per class |
| **Recall** | Measures false negative rate per class |
| **ROC-AUC** | Probability ranking quality across thresholds |
| **Cross-Validation** | 5-fold stratified — prevents overfitting |

### Why Not Accuracy Alone?
If 40% of reviews are positive and a model predicts "positive" for everything, it achieves 40% accuracy — but is completely useless. Macro F1 catches this.

---

## 🧪 Test Results

```
pytest tests/ -v

tests/test_intellisentiment.py::TestPreprocessing::test_basic_cleaning          PASSED
tests/test_intellisentiment.py::TestPreprocessing::test_empty_string_returns_empty PASSED
tests/test_intellisentiment.py::TestPreprocessing::test_stopword_removal        PASSED
tests/test_intellisentiment.py::TestFeatureExtraction::test_tfidf_shape         PASSED
tests/test_intellisentiment.py::TestPrediction::test_single_positive            PASSED
tests/test_intellisentiment.py::TestPrediction::test_batch_prediction           PASSED
tests/test_intellisentiment.py::TestDatabase::test_insert_and_retrieve_prediction PASSED
tests/test_intellisentiment.py::TestFlaskAPI::test_home_returns_200             PASSED
...
```

---

## 🔮 Future Improvements

1. **Deep Learning**: Integrate BERT/RoBERTa for higher accuracy
2. **Aspect-Based Sentiment**: Classify sentiment per product feature
3. **Multilingual Support**: Extend to non-English reviews
4. **Real-Time Dashboard**: WebSocket-based live updates from Kafka
5. **Cloud Deployment**: Docker + Kubernetes + GCP/AWS
6. **Active Learning**: Human-in-the-loop for edge cases
7. **Explainability**: LIME/SHAP for model interpretation
8. **More Data Sources**: Twitter API, Google Reviews, App Store

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 👨‍💻 Author

Built as a comprehensive demonstration of Data Science, Machine Learning Engineering, and NLP Engineering skills.

