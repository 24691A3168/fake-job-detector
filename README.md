# Fake Job Posting Detector - Render Deployment

## Files required
- app.py
- templates/index.html
- requirements.txt
- fraudulent_job_model.keras
- tfidf_vectorizer.pkl
- categorical_preprocessor.pkl

The original training script is included separately as `fake_job_posting_detection.py`.
It is not run by the web service.

## Before deployment
The three model/preprocessor files must exist:
1. fraudulent_job_model.keras
2. tfidf_vectorizer.pkl
3. categorical_preprocessor.pkl

They are created by the training script after it finishes successfully.

## Run locally
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:10000

## Render
Create a Web Service from this GitHub repository.

Build Command:
pip install -r requirements.txt

Start Command:
gunicorn app:app

No database is required.
