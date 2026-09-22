# Phishing Email Detection Model

A beginner-friendly full-stack project using:

- Python Flask backend
- HTML/CSS/JavaScript frontend
- Scikit-learn TF-IDF + Random Forest model
- URL and suspicious-keyword analysis
- Accuracy and confusion matrix

## Run on Windows

Open Command Prompt inside this project folder.

### 1. Create virtual environment

```cmd
py -m venv venv

### 2. Activate it

```cmd
venv\Scripts\activate

### 3. Install packages

```cmd
py -m pip install -r requirements.txt

### 4. Start the backend

```cmd
py app.py

### 5. Open the UI

Go to:

http://127.0.0.1:5000

The first start automatically trains the model and creates `phishing_model.pkl`.

## API

POST `/api/predict`

Example JSON:

```json
{
  "email": "URGENT! Verify your account at http://example.com"
}
```

GET `/api/metrics`

Returns test accuracy and confusion matrix.
