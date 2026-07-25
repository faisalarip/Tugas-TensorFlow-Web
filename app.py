from __future__ import annotations

from pathlib import Path

from flask import Flask, render_template_string, request
import tensorflow as tf
from PIL import Image

from utils.cifar10_utils import preprocess_image


MODEL_PATH = Path("models/cifar10_cnn.keras")
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Klasifikasi Gambar CIFAR-10</title>
</head>
<body>
    <h1>Klasifikasi Gambar CIFAR-10</h1>
    <p>Upload gambar lalu klik prediksi.</p>

    {% if error %}
    <p><strong>Error:</strong> {{ error }}</p>
    {% endif %}

    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
        <button type="submit">Prediksi</button>
    </form>

    {% if prediction %}
    <h2>Hasil Prediksi</h2>
    <p>Kelas: <strong>{{ prediction }}</strong></p>
    <p>Probabilitas: <strong>{{ probability }}</strong></p>
    {% endif %}
</body>
</html>
"""

app = Flask(__name__)
model: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    if not MODEL_PATH.exists():
        return render_template_string(
            HTML_TEMPLATE,
            error="Model belum ditemukan. Jalankan python3 train_cnn.py terlebih dahulu.",
            prediction=None,
            probability=None,
        )

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return render_template_string(
                HTML_TEMPLATE,
                error="Silakan upload gambar terlebih dahulu.",
                prediction=None,
                probability=None,
            )

        image = Image.open(uploaded_file.stream)
        input_tensor = preprocess_image(image)
        probabilities = load_model().predict(input_tensor, verbose=0)[0]
        predicted_index = int(tf.argmax(probabilities))
        predicted_label = CLASS_NAMES[predicted_index]
        predicted_score = float(probabilities[predicted_index]) * 100

        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            prediction=predicted_label,
            probability=f"{predicted_score:.2f}%",
        )

    return render_template_string(
        HTML_TEMPLATE,
        error=None,
        prediction=None,
        probability=None,
    )


if __name__ == "__main__":
    app.run(debug=True)
