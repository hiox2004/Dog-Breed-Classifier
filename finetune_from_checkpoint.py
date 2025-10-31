"""
Fine-tune a saved model by unfreezing top layers and training with a lower learning rate.
Usage:
  python finetune_from_checkpoint.py --model artifacts/best_model.keras --epochs 5 --pct 0.1 --lr 0.00003
"""

import argparse
import os
from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy

from src.config import Config
from src.data import (
    load_labels,
    attach_filepaths,
    split_train_val,
    make_generators,
)


def _get_base_submodel(model: Any, base_model_name: str) -> Any:
    """Return the inserted pre-trained base submodel by name."""
    name = None
    if "mobilenet" in base_model_name.lower():
        name = "mobilenetv2_1.00_224"
    elif "inception" in base_model_name.lower():
        name = "inception_v3"
    if name is None:
        # Fallback: try to find first Functional layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and len(layer.layers) > 10:
                return layer
        raise ValueError("Could not identify base model")
    return model.get_layer(name)


def unfreeze_top_layers(model: Any, base_model_name: str, pct: float) -> int:
    """
    Unfreeze the top `pct` of layers in the base model; returns number of layers unfrozen.
    """
    base = _get_base_submodel(model, base_model_name)
    total = len(base.layers)
    pct = max(0.0, min(1.0, pct))
    k = int(total * pct)
    unfrozen = 0
    for layer in base.layers[-k:]:
        if isinstance(layer, (BatchNormalization,)):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen += 1
    base.trainable = True
    return unfrozen


def fine_tune(cfg: Config, model_path: str, fine_tune_epochs: int, fine_tune_pct: float, fine_tune_lr: float):
    # Load saved model
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Load data
    print("Loading data ...")
    df = load_labels(cfg.LABELS_CSV)
    df = attach_filepaths(df, cfg.TRAIN_DIR)
    train_df, val_df = split_train_val(df, val_size=0.2, seed=cfg.SEED)
    train_gen, val_gen, class_names, class_weights = make_generators(
        train_df,
        val_df,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE,
        seed=cfg.SEED,
        classes=sorted(df["breed"].unique().tolist()),
    )

    # Unfreeze top layers
    print(f"Unfreezing top {fine_tune_pct*100:.0f}% of base model layers ...")
    unfrozen = unfreeze_top_layers(model, cfg.BASE_MODEL_NAME, fine_tune_pct)
    print(f"Unfrozen {unfrozen} layers (BatchNorm layers kept frozen).")

    # Recompile with lower LR
    print(f"Recompiling model with LR={fine_tune_lr} ...")
    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )

    # Callbacks
    ft_best_path = cfg.BEST_MODEL_PATH.replace(".keras", "_finetuned.keras")
    ft_final_path = cfg.FINAL_MODEL_PATH.replace(".keras", "_finetuned.keras")
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
            filepath=ft_best_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        ),
    ]

    # Fine-tune
    print(f"Starting fine-tuning for {fine_tune_epochs} epochs ...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final
    print(f"Saving fine-tuned model to {ft_final_path} ...")
    os.makedirs(os.path.dirname(ft_final_path), exist_ok=True)
    model.save(ft_final_path)

    # Evaluate
    print("Evaluating fine-tuned model on validation set ...")
    val_loss, val_acc, val_top3 = model.evaluate(val_gen, verbose=1)
    print(f"Fine-tuned Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val Top-3: {val_top3:.4f}")

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a saved model by unfreezing top layers.")
    parser.add_argument("--model", default="artifacts/best_model.keras", help="Path to saved model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--pct", type=float, default=0.1, help="Fraction [0-1] of top base layers to unfreeze")
    parser.add_argument("--lr", type=float, default=3e-5, help="Fine-tuning learning rate")
    parser.add_argument("--base_model", default="MobileNetV2", help="Base model name (for layer identification)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    cfg.BASE_MODEL_NAME = args.base_model
    fine_tune(cfg, args.model, args.epochs, args.pct, args.lr)
