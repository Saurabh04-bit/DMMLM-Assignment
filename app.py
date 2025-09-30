import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load trained model
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset to get feature names
data = pd.read_csv("winequality-red.csv")
feature_names = list(data.columns[:-1])  # all except 'quality'

@app.route("/")
def home():
    return {"message": "Wine Quality Prediction API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        input_data = request.get_json()

        # Ensure correct feature order
        features = [input_data.get(feat, 0) for feat in feature_names]

        # Convert to numpy array for prediction
        features_array = np.array(features).reshape(1, -1)

        # Predict using model
        prediction = model.predict(features_array)[0]

        return jsonify({
            "input": input_data,
            "predicted_quality": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

