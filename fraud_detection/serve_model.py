
from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
from utils import preprocess_input

# Load the trained model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(filename='api.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Define route for health check
@app.route("/ping", methods=["GET"])
def ping():
    return "API is alive!"

# Define route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request data
        data = request.json
        logging.info(f"Received request: {data}")

        # Preprocess input data
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        result = {"prediction": int(prediction[0])}  # Convert to Python int

        logging.info(f"Prediction: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
