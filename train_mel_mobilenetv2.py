#!/usr/bin/env python3
"""
Train a MobileNetV2-based classifier on log-mel spectrogram representations
of audio clips organised in UrbanSound-style folders (dataset/<class>/*.wav).

Outputs (metrics, plots, JSON report) follow the conventions used by the
existing train_mfcc_* scripts, but the feature pipeline is log-mel based.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_DIR = Path("dataset")
SAMPLE_RATE = 32_000
CLIP_DURATION = 4.0  # seconds
N_MELS = 128
N_FFT = 1_024
HOP_LENGTH = 256
FMIN = 20
FMAX = 8_000

IMAGE_SIZE = (224, 224)
CHANNELS = 3

TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
RANDOM_SEED = 42

EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

RESULTS_DIR = Path("results/melspectrogram_mobilenetv2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_log_mel(path: Path) -> np.ndarray:
    """Load audio file, truncate/pad to target length, compute log-mel."""
    waveform, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=CLIP_DURATION,
    )
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    return log_mel.astype(np.float32)


def prepare_image(log_mel: np.ndarray) -> np.ndarray:
    """Resize log-mel to IMAGE_SIZE and convert to 3-channel tensor."""
    log_mel = log_mel[np.newaxis, ..., np.newaxis]  # (1, n_mels, time, 1)
    resized = tf.image.resize(log_mel, IMAGE_SIZE, method="bilinear").numpy()[0]
    # Normalise to 0-255 then broadcast to 3 channels
    min_val = resized.min()
    max_val = resized.max()
    norm = (resized - min_val) / (max_val - min_val + 1e-6)
    norm = (norm * 255.0).astype(np.float32)
    image = np.repeat(norm, CHANNELS, axis=-1)
    return image


def load_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return feature tensor (N, H, W, C), label array, and class names."""
    features: List[np.ndarray] = []
    labels: List[str] = []

    class_names = sorted(
        [d.name for d in root.iterdir() if d.is_dir()],
        key=str.lower,
    )
    if not class_names:
        raise RuntimeError(f"No class directories found in {root}")

    print(f"Found classes: {class_names}")

    for class_name in class_names:
        class_dir = root / class_name
        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"Processing {class_name}: {len(wav_files)} files")
        for wav_path in wav_files:
            try:
                log_mel = load_log_mel(wav_path)
                image = prepare_image(log_mel)
                features.append(image)
                labels.append(class_name)
            except Exception as exc:  # pragma: no cover - log and continue
                print(f"Warning: failed to process {wav_path}: {exc}")

    X = np.stack(features)
    y = np.array(labels)
    return X, y, class_names


def build_model(num_classes: int) -> Model:
    """Construct MobileNetV2 transfer learning model."""
    input_layer = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS))
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=input_layer,
    )
    base_model.trainable = False  # freeze backbone

    x = tf.keras.applications.mobilenet_v2.preprocess_input(input_layer)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training(history, out_path: Path) -> None:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: List[str], out_path: Path, normalize: bool = False) -> None:
    plt.figure(figsize=(10, 8))
    data = cm.astype(np.float32)
    if normalize:
        data = data / (data.sum(axis=1, keepdims=True) + 1e-6)
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class_accuracy(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=acc)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Class")
    plt.title("Class-wise Accuracy")
    plt.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(acc):
        plt.text(idx, value + 0.02, f"{value:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_predictions(
    samples: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    num_samples = min(len(samples), len(axes))

    for idx in range(num_samples):
        axes[idx].imshow(samples[idx, :, :, 0], aspect="auto", origin="lower", cmap="magma")
        axes[idx].set_title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        axes[idx].axis("off")

    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Main script
# --------------------------------------------------------------------------- #

def main() -> None:
    set_global_seed(RANDOM_SEED)

    X, y, class_names = load_dataset(DATASET_DIR)
    print(f"Feature tensor shape: {X.shape}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=TEST_SPLIT,
        stratify=y_categorical,
        random_state=RANDOM_SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )

    model = build_model(num_classes=len(class_names))

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss", verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss", verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    # Save results
    model.save(RESULTS_DIR / "mobilenetv2_mel_model.keras")
    plot_training(history, RESULTS_DIR / "training_history.png")
    plot_confusion(cm, class_names, RESULTS_DIR / "confusion_matrix.png", normalize=False)
    plot_confusion(cm, class_names, RESULTS_DIR / "confusion_matrix_normalized.png", normalize=True)
    plot_class_accuracy(cm, class_names, RESULTS_DIR / "class_accuracy.png")
    plot_sample_predictions(
        X_test,
        y_true,
        y_pred,
        class_names,
        RESULTS_DIR / "sample_predictions.png",
    )

    results_payload = {
        "model_type": "MobileNetV2 (frozen) + Dense head (Mel spectrogram)",
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
        "classification_report": report,
    }

    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2, ensure_ascii=False)

    print(f"Training complete. Outputs saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
