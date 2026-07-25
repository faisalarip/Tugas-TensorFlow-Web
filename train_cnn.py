from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

from utils.cifar10_utils import CLASS_NAMES


def build_model() -> tf.keras.Model:
    """Build a compact CNN for CIFAR-10 classification."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            data_augmentation,
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.35),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ],
        name="cifar10_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data() -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """Load and normalize CIFAR-10 data."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def train_model(model_path: Path, epochs: int, batch_size: int) -> None:
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model berhasil disimpan di: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN model for CIFAR-10.")
    parser.add_argument(
        "--model-path",
        default="models/cifar10_cnn.keras",
        help="Lokasi file model hasil training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Jumlah epoch training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Ukuran batch training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(Path(args.model_path), args.epochs, args.batch_size)
