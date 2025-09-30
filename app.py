import pickle
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json(force=True)
        # Assuming the input data is a list of features in the same order as the training data
        prediction = model.predict(np.array([data['features']]))
        # Convert prediction to a standard Python type
        output = int(prediction[0])
        return jsonify(prediction=output)
    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    # This is for local development and testing.
    # For production deployment, a production-ready WSGI server like Gunicorn is recommended.
    app.run(debug=True)
