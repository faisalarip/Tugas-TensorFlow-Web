from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image


CLASS_NAMES: List[str] = [
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

IMAGE_SIZE = (32, 32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize an uploaded image to CIFAR-10 input shape and normalize it."""
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def format_probabilities(probabilities: np.ndarray) -> List[tuple[str, float]]:
    """Pair class names with probabilities sorted from highest to lowest."""
    scores = probabilities.flatten().tolist()
    pairs = list(zip(CLASS_NAMES, scores))
    return sorted(pairs, key=lambda item: item[1], reverse=True)
