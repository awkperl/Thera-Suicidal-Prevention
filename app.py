from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load saved components
model = load_model("Suicidal preventional_model.h5")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(input_vector)
    label_idx = np.argmax(prediction, axis=1)[0]
    label = encoder.inverse_transform([label_idx])[0]
    return label

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run()