from flask import Flask, request, jsonify
from preprocessing import preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# Load model dan tokenizer
model = load_model('model/model_lstm_stress.h5')
with open('model/tokenizer_stress.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100  # sesuai saat training

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text input is required"}), 400

        cleaned_text = preprocess(text)
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(sequence, maxlen=max_len)

        prob = float(model.predict(padded)[0][0])
        prediction = "Negative" if prob < 0.5 else "Positive"
        stress_percent = float(round((1 - prob) * 100, 2))

        return jsonify({
            "prediction": prediction,
            "stress_percent": stress_percent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)