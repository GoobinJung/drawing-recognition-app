import base64
import binascii
import io
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

from model import DrawingCNN


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "drawing_cnn.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

torch.set_num_threads(min(4, os.cpu_count() or 1))

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

CLASS_NAMES = checkpoint["class_names"]

model = DrawingCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def prepare_drawing(image_data):
    prefix = "data:image/png;base64,"

    if not isinstance(image_data, str) or not image_data.startswith(prefix):
        raise ValueError("Please send a PNG drawing.")

    image_bytes = base64.b64decode(
        image_data[len(prefix):],
        validate=True,
    )

    with Image.open(io.BytesIO(image_bytes)) as source:
        if source.format != "PNG":
            raise ValueError("Please send a PNG drawing.")

        if source.width > 1024 or source.height > 1024:
            raise ValueError("The image is too large.")

        rgba = source.convert("RGBA")

    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)

    # Turn black lines on white into white lines on black.
    ink = ImageOps.invert(background.convert("L"))

    box = ink.point(lambda value: 255 if value > 20 else 0).getbbox()

    if box is None:
        raise ValueError("Please draw something first.")

    # Crop, preserve the shape, and center it in a 28×28 image.
    ink = ink.crop(box)
    scale = 24 / max(ink.size)

    new_size = (
        max(1, round(ink.width * scale)),
        max(1, round(ink.height * scale)),
    )
    ink = ink.resize(new_size, Image.Resampling.LANCZOS)

    centered = Image.new("L", (28, 28), 0)
    centered.paste(
        ink,
        ((28 - ink.width) // 2, (28 - ink.height) // 2),
    )

    pixels = np.array(centered, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


@app.get("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(error="Please send a drawing."), 400

    try:
        image = prepare_drawing(payload.get("image"))
    except (ValueError, binascii.Error, OSError):
        return jsonify(
            error="Please draw something on the canvas and try again."
        ), 400

    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)
        scores, indices = probabilities.topk(3, dim=1)

    predictions = [
        {
            "label": CLASS_NAMES[index],
            "confidence": score,
        }
        for score, index in zip(
            scores[0].tolist(),
            indices[0].tolist(),
        )
    ]

    return jsonify(predictions=predictions)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)