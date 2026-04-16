from flask import Flask, request, jsonify
from inference import AlzheimerPredictor

app = Flask(__name__)

predictor = AlzheimerPredictor(
    "final_model.keras",
    "scaler.pkl",
    "feature_names.pkl"
)

@app.route('/')
def home():
    return "Alzheimer's API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predictor.predict(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run()
