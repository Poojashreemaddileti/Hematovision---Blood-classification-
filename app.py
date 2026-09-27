import os
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, flash

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.h5")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASS_LABELS = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")
except Exception as exc:
    print(f"Model loading failed: {exc}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_bytes):
    img = image.load_img(BytesIO(image_bytes), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_LABELS[predicted_idx]
    confidence = float(predictions[predicted_idx] * 100)

    return predicted_class, confidence


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if model is None:
            flash("The AI model could not be loaded. Check the server logs.", "error")
            return render_template("home.html")

        file = request.files.get("file")

        if not file or not file.filename:
            flash("Please choose a blood-cell image first.", "error")
            return render_template("home.html")

        if not allowed_file(file.filename):
            flash("Please upload a JPG, JPEG, PNG or WEBP image.", "error")
            return render_template("home.html")

        image_bytes = file.read()

        if not image_bytes:
            flash("The uploaded image is empty.", "error")
            return render_template("home.html")

        try:
            predicted_class, confidence = predict_image(image_bytes)

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_mime = file.mimetype or "image/jpeg"

            return render_template(
                "result.html",
                predicted_class=predicted_class,
                confidence=confidence,
                image_data_url=f"data:{image_mime};base64,{image_b64}",
                original_filename=file.filename,
            )

        except Exception:
            app.logger.exception("Image prediction failed.")
            flash(
                "The image could not be processed. Please upload a valid blood-cell image.",
                "error",
            )
            return render_template("home.html")

    return render_template("home.html")


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.route("/project-overview")
def project_overview():
    return render_template("project_overview.html")


@app.errorhandler(413)
def too_large(_):
    flash("Image is too large. Maximum upload size is 10 MB.", "error")
    return render_template("home.html"), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
