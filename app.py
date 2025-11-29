# app.py
"""
Flask app to deploy a Wine Quality prediction model.

- Trains (or loads) a model using winequality-red.csv
- Serves a web form at "/" to input feature values
- Returns prediction at "/predict"
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
MODEL_PATH = "wine_model.pkl"       # Saved model file
DATA_PATH = "winequality-red.csv"   # Dataset (must be in same folder)

app = Flask(__name__)


def train_and_save_model():
    """
    Train a RandomForest model on winequality-red.csv
    and save it to disk along with feature names.
    """
    # Your file is comma-separated, so don't force sep=";"
    df = pd.read_csv(DATA_PATH)  # default = comma-separated

    print("Columns in dataset:", list(df.columns))

    # Now the header should become:
    # ['fixed acidity', 'volatile acidity', ..., 'alcohol', 'quality']

    # Use 'quality' as the target column, rest as features
    X = df.drop("quality", axis=1)
    y = df["quality"]

    print("Using features:", list(X.columns))
    print("Using target:", y.name)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    feature_names = list(X.columns)

    joblib.dump((model, feature_names), MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    return model, feature_names


    print("Columns in dataset:", list(df.columns))
    print("Using features:", list(X.columns))
    print("Using target:", y.name)


    # You can tune these hyperparameters if you want
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    feature_names = list(X.columns)

    # Save both the model and feature names
    joblib.dump((model, feature_names), MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    return model, feature_names


def load_model():
    """
    Load model from disk if available, otherwise train a new one.
    """
    if os.path.exists(MODEL_PATH):
        model, feature_names = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model file not found. Training a new model...")
        model, feature_names = train_and_save_model()
    return model, feature_names


# Load (or train) model once when the app starts
model, feature_names = load_model()


@app.route("/", methods=["GET"])
def index():
    """
    Show an input form for all features.
    You should create templates/index.html that loops over feature_names.
    """
    return render_template("index.html", feature_names=feature_names)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive form data, make prediction, and show result.
    """
    try:
        # Extract feature values from the form in the correct order
        values = []
        for name in feature_names:
            raw_val = request.form.get(name)

            if raw_val is None or raw_val.strip() == "":
                return f"Missing value for feature: {name}", 400

            values.append(float(raw_val))

        # Convert to 2D array for scikit-learn: shape (1, n_features)
        input_array = np.array([values])

        # Predict with the loaded model
        predicted_quality = model.predict(input_array)[0]

        paired_data = list(zip(feature_names, values))

        return render_template(
            "result.html",
            prediction=int(predicted_quality),
            paired_data=paired_data
        )
    except Exception as e:
        # Simple error display (you can make this prettier)
        return f"Error during prediction: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

