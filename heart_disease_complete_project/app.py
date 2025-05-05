from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model once at startup
with open("my_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return "Heart Disease Predictor API. Send POST to /predict with a JSON array named 'features'."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array([data["features"]])
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
