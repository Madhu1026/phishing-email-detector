from flask import Flask, render_template, request, jsonify
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

MODEL_FILE = "phishing_model.pkl"
DATA_FILE = os.path.join("dataset", "emails.csv")

def build_model():
    df = pd.read_csv(DATA_FILE).dropna(subset=["text", "label"])
    df["label_num"] = df["label"].map({"Safe": 0, "Phishing": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    report = classification_report(
        y_test, predictions,
        target_names=["Safe", "Phishing"],
        output_dict=True,
        zero_division=0
    )

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "report": report
    }

    joblib.dump(bundle, MODEL_FILE)
    return bundle

def load_model():
    if not os.path.exists(MODEL_FILE):
        return build_model()
    return joblib.load(MODEL_FILE)

bundle = load_model()
model = bundle["model"]
vectorizer = bundle["vectorizer"]

def analyze_email(text):
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    suspicious_words = [
        "urgent", "verify", "password", "account", "suspended",
        "click", "login", "winner", "prize", "confirm",
        "bank", "payment", "security"
    ]

    lower = text.lower()
    found_keywords = [word for word in suspicious_words if word in lower]

    features = vectorizer.transform([text])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    label = "Phishing" if prediction == 1 else "Safe"
    confidence = float(probabilities[prediction])

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "email_length": len(text),
        "url_count": len(urls),
        "urls": urls,
        "keyword_count": len(found_keywords),
        "keywords": found_keywords,
        "exclamation_count": text.count("!")
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()

    if not email:
        return jsonify({"error": "Please enter email content."}), 400

    return jsonify(analyze_email(email))

@app.route("/api/metrics")
def metrics():
    return jsonify({
        "accuracy": round(bundle["accuracy"] * 100, 2),
        "confusion_matrix": bundle["confusion_matrix"],
        "report": bundle["report"]
    })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
