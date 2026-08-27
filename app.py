from flask import Flask, render_template, request
import pandas as pd
import re
import joblib
import nltk
from tensorflow.keras.models import load_model
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Download required NLTK resources at startup
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

model = load_model("fraudulent_job_model.keras")
tfidf = joblib.load("tfidf_vectorizer.pkl")
preprocessor = joblib.load("categorical_preprocessor.pkl")

text_columns = ["title", "company_profile", "description", "requirements", "benefits"]
cat_columns = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function"
]
binary_columns = ["telecommuting", "has_company_logo", "has_questions"]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

def predict_job(job_dict):
    df_new = pd.DataFrame([job_dict])

    for col in text_columns:
        if col not in df_new:
            df_new[col] = ""
        df_new[col] = df_new[col].fillna("")

    for col in cat_columns:
        if col not in df_new:
            df_new[col] = "missing"
        df_new[col] = df_new[col].fillna("missing")

    for col in binary_columns:
        if col not in df_new:
            df_new[col] = 0
        df_new[col] = df_new[col].fillna(0)

    df_new["full_text"] = df_new[text_columns].agg(" ".join, axis=1)
    df_new["cleaned_text"] = df_new["full_text"].apply(clean_text)

    text_vec = tfidf.transform(df_new["cleaned_text"])
    cat_vec = preprocessor.transform(df_new[cat_columns + binary_columns])
    X_new = hstack([text_vec, cat_vec]).toarray()

    probability = float(model.predict(X_new, verbose=0)[0][0])
    prediction = "FRAUDULENT" if probability > 0.5 else "REAL"
    return probability, prediction

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    job = {
        "title": request.form.get("title", ""),
        "location": request.form.get("location", "missing"),
        "department": request.form.get("department", "missing"),
        "salary_range": request.form.get("salary_range", "missing"),
        "company_profile": request.form.get("company_profile", ""),
        "description": request.form.get("description", ""),
        "requirements": request.form.get("requirements", ""),
        "benefits": request.form.get("benefits", ""),
        "telecommuting": int(request.form.get("telecommuting", 0)),
        "has_company_logo": int(request.form.get("has_company_logo", 0)),
        "has_questions": int(request.form.get("has_questions", 0)),
        "employment_type": request.form.get("employment_type", "missing"),
        "required_experience": request.form.get("required_experience", "missing"),
        "required_education": request.form.get("required_education", "missing"),
        "industry": request.form.get("industry", "missing"),
        "function": request.form.get("function", "missing"),
    }

    try:
        probability, prediction = predict_job(job)
        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability * 100, 2)
        )
    except Exception as e:
        return render_template("index.html", error=str(e)), 500

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
