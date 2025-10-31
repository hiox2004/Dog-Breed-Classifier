import json
import os
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_labels(labels_csv: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if not {"id", "breed"}.issubset(df.columns):
        raise ValueError("labels.csv must contain 'id' and 'breed' columns")
    return df


def attach_filepaths(df: pd.DataFrame, train_dir: str) -> pd.DataFrame:
    df = df.copy()
    df["filename"] = df["id"].apply(lambda x: os.path.join(train_dir, f"{x}.jpg"))
    # Filter out any missing files to avoid generator errors
    exists_mask = df["filename"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"Warning: {missing} image files listed in labels.csv were not found in '{train_dir}'. They will be skipped.")
    df = df[exists_mask].reset_index(drop=True)
    return df


def plot_class_distribution(df: pd.DataFrame, out_path: str, top_n: int = None) -> None:
    counts = df["breed"].value_counts()
    plt.figure(figsize=(20, 6))
    if top_n is not None and top_n < len(counts):
        counts.head(top_n).plot(kind="bar")
        plt.title(f"Top {top_n} breeds by count")
    else:
        counts.plot(kind="bar")
        plt.title("Class distribution (all breeds)")
    plt.xlabel("Breed")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def split_train_val(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["breed"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame, classes: List[str]) -> Dict[int, float]:
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=train_df["breed"].values,
    )
    # Map class label -> index according to provided classes order
    class_to_index = {c: i for i, c in enumerate(classes)}
    # Return as {index: weight}
    return {class_to_index[c]: float(w) for c, w in zip(classes, weights)}


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    classes: List[str] = None,
):
    """
    Create Keras ImageDataGenerator-based generators with augmentation for training and
    simple rescaling for validation. Ensures consistent class order via `classes`.
    Returns (train_gen, val_gen, class_names, class_weights)
    """
    if classes is None:
        # Sort to enforce deterministic order
        classes = sorted(train_df["breed"].unique().tolist())

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # We pass absolute filepaths, so directory=None
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="breed",
        classes=classes,
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validate_filenames=False,
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="breed",
        classes=classes,
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
        validate_filenames=False,
    )

    # Ensure class order is consistent and returned
    class_indices = train_gen.class_indices  # dict: label -> index
    # Convert to index-ordered list of class names
    class_names = [None] * len(class_indices)
    for label, idx in class_indices.items():
        class_names[idx] = label

    # Compute class weights to mitigate imbalance
    class_weights = compute_class_weights(train_df, class_names)

    return train_gen, val_gen, class_names, class_weights


def save_class_names(class_names: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
