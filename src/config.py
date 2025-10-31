from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # Paths
    LABELS_CSV: str = "data/labels.csv"
    TRAIN_DIR: str = "data/train"
    ARTIFACTS_DIR: str = "artifacts"

    # Data / Model
    IMAGE_SIZE: Tuple[int, int] = (299, 299)
    BATCH_SIZE: int = 32
    SEED: int = 42
    NUM_CLASSES: int = 120  # Expected classes; validated from labels

    # Base model: 'MobileNetV2' or 'InceptionV3'
    BASE_MODEL_NAME: str = "InceptionV3"

    # Training
    EPOCHS: int = 15
    LEARNING_RATE: float = 1e-3
    DROPOUT_RATE: float = 0.3
    DENSE_UNITS: int = 512
    
    # Fine-tuning
    FINE_TUNE: bool = True
    FINE_TUNE_PCT: float = 0.3  # Unfreeze top 30% of base layers
    FINE_TUNE_EPOCHS: int = 5
    FINE_TUNE_LR: float = 1e-4

    # Callbacks
    EARLY_STOPPING_PATIENCE: int = 3
    REDUCE_LR_PATIENCE: int = 2

    # Output files
    BEST_MODEL_PATH: str = "artifacts/best_model.keras"
    FINAL_MODEL_PATH: str = "artifacts/final_model.keras"
    CLASSES_JSON_PATH: str = "artifacts/classes.json"
    CLASS_DISTRIBUTION_PNG: str = "artifacts/class_distribution.png"
    CONFUSION_MATRIX_PNG: str = "artifacts/confusion_matrix.png"
