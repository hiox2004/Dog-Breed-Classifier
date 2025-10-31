import argparse
import json
from typing import List, Tuple

import numpy as np
import tensorflow as tf


def load_classes(classes_json_path: str) -> List[str]:
    with open(classes_json_path, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return classes


def preprocess_image(image_path: str, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image into a 4D batch (1, H, W, 3) normalized to [0,1]."""
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def top_k_predictions(probs: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """Return top-k (class_name, confidence) from a single-probability row."""
    idxs = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idxs]


class DogBreedPredictor:
    def __init__(self, model_path: str, classes_json_path: str, image_size: Tuple[int, int] = (224, 224)):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = load_classes(classes_json_path)
        self.image_size = image_size

    def predict_top_k(self, image_path: str, k: int = 3) -> List[Tuple[str, float]]:
        batch = preprocess_image(image_path, self.image_size)
        probs = self.model.predict(batch, verbose=0)[0]
        return top_k_predictions(probs, self.classes, k)


def main():
    parser = argparse.ArgumentParser(description="Predict top-3 dog breeds for an input image.")
    parser.add_argument("--image", required=True, help="Path to the input dog image")
    parser.add_argument("--model", default="artifacts/best_model.keras", help="Path to trained Keras model")
    parser.add_argument("--classes", default="artifacts/classes.json", help="Path to classes.json file")
    parser.add_argument("--image_size", default="224,224", help="Image size as H,W (e.g., 224,224)")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions to show")
    args = parser.parse_args()

    h, w = map(int, args.image_size.split(","))
    predictor = DogBreedPredictor(args.model, args.classes, (h, w))
    preds = predictor.predict_top_k(args.image, args.topk)

    print("Top predictions:")
    for rank, (breed, conf) in enumerate(preds, 1):
        print(f"{rank}. {breed}: {conf:.4f}")


if __name__ == "__main__":
    main()
