from typing import Tuple

import tensorflow as tf


def build_model(
    num_classes: int,
    image_size: Tuple[int, int] = (224, 224),
    base_model_name: str = "MobileNetV2",
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    dense_units: int = 512,
) -> tf.keras.Model:
    """
    Build a transfer learning model with a chosen ImageNet base (frozen) and a custom head.
    Supported base_model_name: 'MobileNetV2', 'InceptionV3'
    """
    h, w = image_size

    # Input layer (we normalize in the generators to [0,1])
    inputs = tf.keras.Input(shape=(h, w, 3), name="input_image")

    base_model_name = base_model_name.lower()
    if base_model_name == "mobilenetv2":
        Base = tf.keras.applications.MobileNetV2
    elif base_model_name == "inceptionv3":
        Base = tf.keras.applications.InceptionV3
    else:
        raise ValueError("Unsupported base_model_name. Use 'MobileNetV2' or 'InceptionV3'.")

    base_model = Base(include_top=False, weights="imagenet", input_shape=(h, w, 3))
    base_model.trainable = False  # Freeze for initial training

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{Base.__name__}_dogbreed")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model
