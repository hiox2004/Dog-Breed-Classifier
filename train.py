import json
import os
import argparse
from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.data import (
    load_labels,
    attach_filepaths,
    plot_class_distribution,
    split_train_val,
    make_generators,
    save_class_names,
)
from src.model import build_model


def _get_base_submodel(model: Any, base_model_name: str) -> Any:
    """Return the inserted pre-trained base submodel by name."""
    name = None
    if base_model_name.lower() == "mobilenetv2":
        name = "mobilenetv2_1.00_224"
    elif base_model_name.lower() == "inceptionv3":
        name = "inceptionv3"
    if name is None:
        raise ValueError("Unsupported base model name")
    return model.get_layer(name)


def unfreeze_top_layers(model: Any, base_model_name: str, pct: float) -> int:
    """
    Unfreeze the top `pct` of layers in the base model; returns number of layers unfrozen.
    """
    base = _get_base_submodel(model, base_model_name)
    total = len(base.layers)
    pct = max(0.0, min(1.0, pct))
    k = int(total * pct)
    # Keep batchnorm layers typically frozen for stability
    unfrozen = 0
    for layer in base.layers[-k:]:
        if isinstance(layer, (BatchNormalization,)):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    base.trainable = True  # ensure container reflects inner trainable flags
    return unfrozen


def train_and_evaluate(cfg: Config):
    # 1) Data Preprocessing & EDA
    print("Loading labels.csv ...")
    df = load_labels(cfg.LABELS_CSV)

    print("Attaching file paths ...")
    df = attach_filepaths(df, cfg.TRAIN_DIR)

    # Basic checks
    n_classes = df["breed"].nunique()
    print(f"Unique breeds in labels: {n_classes}")
    if cfg.NUM_CLASSES and n_classes != cfg.NUM_CLASSES:
        print(f"Warning: Expected {cfg.NUM_CLASSES} classes, found {n_classes} in labels.csv.")

    # EDA: class imbalance plot
    print("Saving class distribution plot ...")
    plot_class_distribution(df, cfg.CLASS_DISTRIBUTION_PNG)

    # Train/Val split (stratified)
    print("Splitting train/validation sets ...")
    train_df, val_df = split_train_val(df, val_size=0.2, seed=cfg.SEED)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # 2) Data Generators with augmentation
    print("Creating data generators ...")
    train_gen, val_gen, class_names, class_weights = make_generators(
        train_df,
        val_df,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        classes=sorted(df["breed"].unique().tolist()),
    )

    # Persist class names (index order)
    print("Saving class names mapping ...")
    save_class_names(class_names, cfg.CLASSES_JSON_PATH)

    # 3) Model Architecture (Transfer Learning)
    print(f"Building model with base: {cfg.BASE_MODEL_NAME} ...")
    model = build_model(
        num_classes=len(class_names),
        image_size=cfg.IMAGE_SIZE,
        base_model_name=cfg.BASE_MODEL_NAME,
        learning_rate=cfg.LEARNING_RATE,
        dropout_rate=cfg.DROPOUT_RATE,
        dense_units=cfg.DENSE_UNITS,
    )
    model.summary()

    # 4) Training setup
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=1e-6,
        ),
        ModelCheckpoint(
            filepath=cfg.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    print("Starting training ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Optional fine-tuning phase
    if cfg.FINE_TUNE and cfg.FINE_TUNE_EPOCHS > 0:
        print("\nStarting fine-tuning phase ...")
        unfrozen = unfreeze_top_layers(model, cfg.BASE_MODEL_NAME, cfg.FINE_TUNE_PCT)
        print(f"Unfrozen top layers in base model: {unfrozen}")

        # Re-compile with lower LR
        model.compile(
            optimizer=Adam(learning_rate=cfg.FINE_TUNE_LR),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                TopKCategoricalAccuracy(k=3, name="top3_acc"),
            ],
        )

        ft_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=cfg.FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )
        # Merge histories if needed
        history.history.update({f"ft_{k}": v for k, v in ft_history.history.items()})

    # Save final model
    print("Saving final model ...")
    os.makedirs(os.path.dirname(cfg.FINAL_MODEL_PATH), exist_ok=True)
    model.save(cfg.FINAL_MODEL_PATH)

    # 4) Evaluation
    print("Evaluating on validation set ...")
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=1)
    print(f"Validation Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val Top-3: {val_top3:.4f}")

    # Classification report and confusion matrix
    print("Generating classification report and confusion matrix ...")
    y_true = val_gen.classes
    y_prob = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(cfg.CONFUSION_MATRIX_PNG)
    plt.close()

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train dog breed classifier with transfer learning.")
    parser.add_argument("--base_model", choices=["MobileNetV2", "InceptionV3"], help="Override base model name")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--image_size", type=str, help="Image size as H,W (e.g., 224,224)")
    # fine-tune args
    parser.add_argument("--fine_tune", type=str, choices=["true", "false"], help="Enable/disable fine-tuning phase")
    parser.add_argument("--fine_tune_pct", type=float, help="Fraction [0-1] of top base layers to unfreeze")
    parser.add_argument("--fine_tune_epochs", type=int, help="Number of fine-tuning epochs")
    parser.add_argument("--fine_tune_lr", type=float, help="Learning rate for fine-tuning phase")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    if args.base_model:
        cfg.BASE_MODEL_NAME = args.base_model
    if args.epochs:
        cfg.EPOCHS = args.epochs
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.image_size:
        h, w = map(int, args.image_size.split(","))
        cfg.IMAGE_SIZE = (h, w)
    # apply fine-tune overrides
    if args.fine_tune:
        cfg.FINE_TUNE = args.fine_tune.lower() == "true"
    if args.fine_tune_pct is not None:
        cfg.FINE_TUNE_PCT = args.fine_tune_pct
    if args.fine_tune_epochs is not None:
        cfg.FINE_TUNE_EPOCHS = args.fine_tune_epochs
    if args.fine_tune_lr is not None:
        cfg.FINE_TUNE_LR = args.fine_tune_lr
    train_and_evaluate(cfg)
