# IntelliSentiment AI
### NLP-Based Sentiment Analysis System

An end-to-end sentiment analysis project that uses Natural Language Processing (NLP) and Machine Learning to classify text reviews as **Positive, Negative, or Neutral**. It includes a Flask web application, a prediction interface, and a dashboard for viewing sentiment statistics.

## Project Overview

IntelliSentiment AI demonstrates how machine learning can be used to understand the sentiment expressed in text reviews. The project provides a web interface where users can submit reviews and view sentiment predictions.

The application uses multiple machine learning classifiers and selects a model through the training pipeline.

## Key Features

- **Sentiment Classification:** Classifies text into Positive, Negative, and Neutral categories.
- **Text Preprocessing:** Prepares text for machine learning using NLP techniques.
- **Machine Learning Models:** Includes Logistic Regression, Naive Bayes, and Support Vector Machine (SVM).
- **Model Training Pipeline:** Trains and evaluates the available classifiers.
- **Flask Web Application:** Provides a browser-based interface for sentiment prediction.
- **Interactive Dashboard:** Displays sentiment statistics, trends, and recent predictions.
- **SQLite Database:** Stores prediction records for dashboard use.
- **Automated Testing:** Includes a test suite to check project functionality.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| NLP | NLTK, TF-IDF |
| Machine Learning | Scikit-learn |
| Web Framework | Flask |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Visualization | Chart.js |
| Testing | Pytest |

## Project Architecture

1. **Input:** User submits a text review.
2. **Preprocessing:** Text is prepared for model prediction.
3. **Feature Extraction:** TF-IDF converts text into numerical features.
4. **Classification:** Machine learning models predict the sentiment.
5. **Web Application:** Flask presents the prediction to the user.
6. **Database:** Prediction records are stored in SQLite.
7. **Dashboard:** Displays available sentiment statistics and prediction history.

## Project Structure

```text
IntelliSentiment-AI/
│
├── database/
│   └── intellisentiment.db
│
├── models/
│   └── best_model.joblib
│
├── src/
│   └── train.py
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│
├── tests/
│   └── test_intellisentiment.py
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: This is a simplified overview. The repository contains the complete project files.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Priti356/NLP-Based-Sentiment-Analysis-System.git
```

### 2. Open the project directory

```bash
cd NLP-Based-Sentiment-Analysis-System
```

### 3. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

### 4. Activate the virtual environment (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the training pipeline

```bash
python run_pipeline.py
```

### 7. Start the Flask application

```bash
python app.py
```

### 8. Open the application

Visit:

```text
http://127.0.0.1:5000
```

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

**Verified test result:**

- 30 tests passed
- Test execution completed in approximately 47 seconds
- No test failures in the recorded run

## Current Project Verification

The following functionality has been tested during development:

| Feature | Status |
|---|---|
| Python training pipeline | Tested successfully |
| Flask application startup | Tested successfully |
| Single-review sentiment prediction | Tested successfully |
| Dashboard loading and updates | Tested successfully |
| SQLite database initialization | Tested successfully |
| Automated test suite | 30 tests passed |
| Bulk CSV prediction | Not yet verified |
| Separate API endpoint testing | Not yet verified |
| Spark, Kafka, and Airflow integrations | Not yet verified |

## Model Evaluation

The training pipeline reported the following results during the recorded run:

| Metric | Reported Result |
|---|---:|
| Accuracy | 1.0000 |
| Macro Precision | 1.0000 |
| Macro Recall | 1.0000 |
| Macro F1-Score | 1.0000 |
| ROC-AUC | 1.0000 |

**Important:** These are results from the current project run, not evidence of guaranteed real-world performance. The repository's default dataset may be synthetic. Further evaluation on an independent, representative dataset is needed before drawing conclusions about model generalization.

## Future Improvements

- Validate model performance on real-world review datasets.
- Improve model evaluation and error analysis.
- Verify and extend bulk CSV prediction.
- Test and document API endpoints.
- Explore transformer-based sentiment classification.
- Add multilingual sentiment analysis.
- Explore explainable AI techniques.

## Learning Outcomes

Through this project, I worked on:

- Natural Language Processing and text preprocessing.
- TF-IDF feature extraction.
- Machine learning classification algorithms.
- Model training and evaluation.
- Building a Flask-based web application.
- Integrating SQLite for prediction records.
- Running automated tests using Pytest.

## License

Refer to the repository's license file for licensing information.

## Repository

[View IntelliSentiment AI on GitHub](https://github.com/Priti356/NLP-Based-Sentiment-Analysis-System)

