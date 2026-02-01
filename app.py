from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load ML model ONCE at startup
# -----------------------------
try:
    model = tf.keras.models.load_model("chest_xray_densenet_model.keras")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed")
    print(e)
    raise e


# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check file
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image_bytes = file.read()

        # Preprocess
        image = preprocess_image(image_bytes)

        # Model inference
        prediction = model.predict(image)
        score = float(prediction[0][0])

        # Binary classification logic
        label = "Pneumonia" if score > 0.5 else "Normal"
        confidence = score if score > 0.5 else (1 - score)

        # Response expected by frontend
        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception:
        print("❌ Prediction error")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
