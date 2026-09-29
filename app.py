import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the pre-trained model into the application memory
model = joblib.load("house_model.pkl")


@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Get the number of rooms from the URL parameter (e.g., /predict?rooms=3)
        rooms = float(request.args.get("rooms", 1))

        # Pass the input to the model for inference
        prediction = model.predict([[rooms]])[0]

        # Return the response as JSON
        return jsonify({"rooms": rooms, "predicted_price_thousands": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
