from __future__ import annotations

import base64
from io import BytesIO
import os
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
<html lang="id">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Klasifikasi Gambar CIFAR-10</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f4f7fb;
            --card: #ffffff;
            --text: #14213d;
            --muted: #5f6b85;
            --primary: #3b82f6;
            --primary-dark: #2563eb;
            --success-bg: #ecfdf3;
            --success-text: #166534;
            --error-bg: #fef2f2;
            --error-text: #b91c1c;
            --border: #dbe3f0;
            --shadow: 0 20px 45px rgba(20, 33, 61, 0.08);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #eef4ff 0%, #f9fbff 100%);
            color: var(--text);
        }

        .wrapper {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px 16px;
        }

        .card {
            width: 100%;
            max-width: 920px;
            background: var(--card);
            border-radius: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .hero {
            padding: 28px 32px 12px;
            background: linear-gradient(135deg, #1d4ed8 0%, #60a5fa 100%);
            color: #fff;
        }

        .hero h1 {
            margin: 0 0 10px;
            font-size: 32px;
        }

        .hero p {
            margin: 0;
            line-height: 1.6;
            max-width: 700px;
        }

        .content {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 24px;
            padding: 28px 32px 32px;
        }

        .section-title {
            margin: 0 0 14px;
            font-size: 20px;
        }

        .upload-box {
            border: 2px dashed #b9cae9;
            border-radius: 16px;
            padding: 20px;
            background: #f8fbff;
        }

        .upload-box p {
            margin-top: 0;
            color: var(--muted);
        }

        input[type="file"] {
            width: 100%;
            margin-bottom: 14px;
        }

        button {
            border: 0;
            background: var(--primary);
            color: #fff;
            padding: 12px 18px;
            border-radius: 10px;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: var(--primary-dark);
        }

        .message {
            padding: 14px 16px;
            border-radius: 12px;
            margin-bottom: 18px;
            font-size: 15px;
        }

        .message.error {
            background: var(--error-bg);
            color: var(--error-text);
        }

        .result-box {
            background: var(--success-bg);
            color: var(--success-text);
            border-radius: 16px;
            padding: 18px;
            margin-bottom: 18px;
        }

        .result-box h3 {
            margin: 0 0 10px;
        }

        .preview-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
        }

        .preview-box img {
            width: 100%;
            max-height: 340px;
            object-fit: contain;
            border-radius: 12px;
            background: #fff;
            border: 1px solid var(--border);
        }

        .helper {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
            margin-top: 12px;
        }

        .class-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 14px;
        }

        .badge {
            padding: 8px 12px;
            border-radius: 999px;
            background: #e8f0ff;
            color: #1d4ed8;
            font-size: 13px;
        }

        @media (max-width: 760px) {
            .content {
                grid-template-columns: 1fr;
                padding: 20px;
            }

            .hero {
                padding: 24px 20px 12px;
            }

            .hero h1 {
                font-size: 26px;
            }
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <main class="card">
            <section class="hero">
                <h1>Klasifikasi Gambar CIFAR-10</h1>
                <p>
                    Website sederhana untuk tugas CNN. Upload gambar objek,
                    lalu sistem akan memprediksi kelas gambar berdasarkan model CIFAR-10.
                </p>
            </section>

            <section class="content">
                <div>
                    <h2 class="section-title">Upload Gambar</h2>

                    {% if error %}
                    <div class="message error"><strong>Error:</strong> {{ error }}</div>
                    {% endif %}

                    {% if prediction %}
                    <div class="result-box">
                        <h3>Hasil Prediksi</h3>
                        <p><strong>Kelas:</strong> {{ prediction }}</p>
                        <p><strong>Probabilitas:</strong> {{ probability }}</p>
                    </div>
                    {% endif %}

                    <div class="upload-box">
                        <p>Pilih file gambar format JPG, JPEG, atau PNG.</p>
                        <form method="post" enctype="multipart/form-data">
                            <input type="file" name="image" accept=".jpg,.jpeg,.png" required>
                            <button type="submit">Prediksi Sekarang</button>
                        </form>
                    </div>

                    <div class="helper">
                        Model mendukung 10 kelas CIFAR-10:
                        <div class="class-list">
                            {% for class_name in class_names %}
                            <span class="badge">{{ class_name }}</span>
                            {% endfor %}
                        </div>
                    </div>
                </div>

                <div>
                    <h2 class="section-title">Preview</h2>
                    <div class="preview-box">
                        {% if image_data %}
                        <img src="data:{{ image_mime }};base64,{{ image_data }}" alt="Preview gambar">
                        {% else %}
                        <img src="https://coresg-normal.trae.ai/api/ide/v1/text_to_image?prompt=minimal%20illustration%20of%20a%20computer%20vision%20image%20classification%20dashboard%2C%20clean%20web%20app%20mockup%2C%20soft%20blue%20background%2C%20professional%20academic%20style&image_size=landscape_4_3" alt="Ilustrasi klasifikasi gambar">
                        {% endif %}
                        <p class="helper">
                            {% if image_data %}
                            Gambar yang Anda upload akan ditampilkan di sini sebelum atau sesudah prediksi.
                            {% else %}
                            Preview gambar akan muncul di area ini setelah file dipilih dan dikirim.
                            {% endif %}
                        </p>
                    </div>
                </div>
            </section>
        </main>
    </div>
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
            image_data=None,
            image_mime=None,
            class_names=CLASS_NAMES,
        )

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            return render_template_string(
                HTML_TEMPLATE,
                error="Silakan upload gambar terlebih dahulu.",
                prediction=None,
                probability=None,
                image_data=None,
                image_mime=None,
                class_names=CLASS_NAMES,
            )

        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        input_tensor = preprocess_image(image)
        probabilities = load_model().predict(input_tensor, verbose=0)[0]
        predicted_index = int(tf.argmax(probabilities))
        predicted_label = CLASS_NAMES[predicted_index]
        predicted_score = float(probabilities[predicted_index]) * 100
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_mime = uploaded_file.mimetype or "image/png"

        return render_template_string(
            HTML_TEMPLATE,
            error=None,
            prediction=predicted_label,
            probability=f"{predicted_score:.2f}%",
            image_data=image_data,
            image_mime=image_mime,
            class_names=CLASS_NAMES,
        )

    return render_template_string(
        HTML_TEMPLATE,
        error=None,
        prediction=None,
        probability=None,
        image_data=None,
        image_mime=None,
        class_names=CLASS_NAMES,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
