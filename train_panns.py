#!/usr/bin/env python3
"""
Train a PANNs (CNN14) feature-extractor + shallow classifier pipeline on an
UrbanSound-like dataset and export results compatible with the existing
train_mfcc outputs.

The script loads waveforms, obtains CNN14 embeddings (optionally cached),
trains a chosen classifier (logistic regression, MLP, optional LightGBM),
evaluates the model, and saves figures and metrics under results/panns_cnn14/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import subprocess
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning)


class SpecAugment:
    """Simple time/frequency masking for spectrogram-like tensors."""

    def __init__(self, time_mask: int = 32, freq_mask: int = 8, p: float = 0.5) -> None:
        self.time_mask = time_mask
        self.freq_mask = freq_mask
        self.p = p

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        # Convert to mel spectrogram, apply masks, invert approximately.
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=128,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mask = mel_db.copy()
        if np.random.rand() < self.p:
            t = np.random.randint(0, mask.shape[1] - self.time_mask + 1)
            mask[:, t : t + self.time_mask] = mask.min()
        if np.random.rand() < self.p:
            f = np.random.randint(0, mask.shape[0] - self.freq_mask + 1)
            mask[f : f + self.freq_mask, :] = mask.min()
        restored = librosa.feature.inverse.mel_to_audio(
            librosa.db_to_power(mask),
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
        )
        return _pad_truncate(restored, len(waveform))


def _pad_truncate(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if len(waveform) == target_len:
        return waveform
    if len(waveform) > target_len:
        return waveform[:target_len]
    pad = target_len - len(waveform)
    return np.pad(waveform, (0, pad))


def _mix_with_noise(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(waveform**2) + 1e-12)
    snr_linear = 10 ** (snr_db / 20)
    noise_rms = rms_signal / snr_linear
    noise = np.random.normal(0.0, 1.0, size=waveform.shape).astype(np.float32)
    current_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    scaled_noise = noise * (noise_rms / current_rms)
    return waveform + scaled_noise.astype(np.float32)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class AudioSample:
    path: Path
    label: str


def discover_dataset(dataset_path: Path) -> Tuple[List[AudioSample], List[str]]:
    samples: List[AudioSample] = []
    classes = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    for cls in classes:
        class_dir = dataset_path / cls
        for wav_path in sorted(class_dir.glob("*.wav")):
            samples.append(AudioSample(path=wav_path, label=cls))
    if not samples:
        raise RuntimeError(f"Dataset boş: {dataset_path}")
    return samples, classes


def load_waveform(
    path: Path,
    sr: int,
    duration: float,
) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    return _pad_truncate(waveform.astype(np.float32), target_len)


def apply_augmentation(
    waveform: np.ndarray,
    sr: int,
    augment: str,
    snr_values: Sequence[float],
    spec_aug: Optional[SpecAugment],
) -> np.ndarray:
    if augment == "none":
        return waveform
    if augment == "noise" and snr_values:
        snr = float(np.random.choice(snr_values))
        return _mix_with_noise(waveform, snr)
    if augment == "spec" and spec_aug is not None:
        return spec_aug(waveform, sr)
    return waveform


def load_panns_cnn14(sample_rate: int, freeze: bool = True) -> nn.Module:
    labels_path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not labels_path.exists() or labels_path.stat().st_size < 10_000:
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading AudioSet label metadata...")
        url = "https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        with urllib.request.urlopen(url) as response, open(labels_path, "wb") as f:
            f.write(response.read())

    try:
        from panns_inference.models import Cnn14
    except ImportError:
        hub_dir = Path(torch.hub.get_dir()) / "qiuqiangkong_panns_inference_master"
        if not hub_dir.exists():
            hub_dir.parent.mkdir(parents=True, exist_ok=True)
            zip_path = hub_dir.parent / "panns_inference_master.zip"
            url = "https://github.com/qiuqiangkong/panns-inference/archive/master.zip"
            print("Downloading panns_inference repository...")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(hub_dir.parent)
            extracted = hub_dir.parent / "panns-inference-master"
            if extracted.exists():
                extracted.rename(hub_dir)
            zip_path.unlink(missing_ok=True)
        if str(hub_dir) not in sys.path:
            sys.path.insert(0, str(hub_dir))
        if str(hub_dir / "panns_inference") not in sys.path:
            sys.path.insert(0, str(hub_dir / "panns_inference"))
        import importlib

        importlib.invalidate_caches()
        try:
            module = importlib.import_module("panns_inference.models")
        except ModuleNotFoundError as exc:
            if exc.name == "torchlibrosa":
                print("Installing torchlibrosa...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "torchlibrosa"])
                module = importlib.import_module("panns_inference.models")
            else:  # pragma: no cover
                raise RuntimeError(
                    f"panns_inference paketini yükleyemedim. sys.path={sys.path[:5]}"
                ) from exc
        Cnn14 = getattr(module, "Cnn14")
    else:
        try:
            import torchlibrosa  # type: ignore
        except ImportError:
            print("Installing torchlibrosa...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchlibrosa"])
            import torchlibrosa  # type: ignore

    checkpoint_path = Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 3e8:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading CNN14 checkpoint from Zenodo...")
        url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
        with urllib.request.urlopen(url) as response, open(checkpoint_path, "wb") as f:
            f.write(response.read())

    model = Cnn14(
        sample_rate=sample_rate,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14_000,
        classes_num=527,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


def get_finetune_layers(model: nn.Module, last_k: int) -> List[nn.Parameter]:
    layers: List[nn.Parameter] = []
    if last_k <= 0:
        return layers
    params = [p for p in model.parameters()]
    for param in params[-last_k:]:
        param.requires_grad = True
        layers.append(param)
    return layers


def extract_embeddings(
    model: nn.Module,
    waveforms: Sequence[np.ndarray],
    sr: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    model = model.to(device)
    model.eval()
    for idx in range(0, len(waveforms), batch_size):
        batch = waveforms[idx : idx + batch_size]
        tensor = torch.from_numpy(np.stack(batch)).float().to(device)
        with torch.no_grad():
            outputs = model(tensor)
            embedding = outputs["embedding"]
            if isinstance(embedding, tuple):
                embedding = embedding[0]
            embedding = embedding.detach().cpu().numpy()
            if embedding.ndim == 3:
                mean_pool = embedding.mean(axis=2)
                std_pool = embedding.std(axis=2)
                embedding = np.concatenate([mean_pool, std_pool], axis=1)
            embeddings.append(embedding)
    return np.concatenate(embeddings, axis=0)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int], num_classes: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev = width
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    class_names: Sequence[str],
    hidden_layers: Sequence[int],
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    class_weights: Optional[np.ndarray],
) -> Tuple[MLPClassifier, Dict[str, List[float]]]:
    num_classes = len(class_names)
    model = MLPClassifier(train_x.shape[1], hidden_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).float().to(device) if class_weights is not None else None
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val = float("inf")
    best_state: Dict[str, Any] = {}
    patience = 10
    epochs_no_improve = 0

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x).float(), torch.from_numpy(train_y).long()
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_x).float(), torch.from_numpy(val_y).long()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}"
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_logreg(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    class_weight: Optional[str],
) -> Tuple[LogisticRegression, Dict[str, List[float]]]:
    clf = LogisticRegression(
        max_iter=10_000,
        n_jobs=-1,
        solver="lbfgs",
        class_weight=class_weight,
    )
    clf.fit(train_x, train_y)
    train_preds = clf.predict(train_x)
    val_preds = clf.predict(val_x)
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [accuracy_score(train_y, train_preds)],
        "val_acc": [accuracy_score(val_y, val_preds)],
    }
    print(
        f"LogReg train_acc={history['train_acc'][0]:.3f} | val_acc={history['val_acc'][0]:.3f}"
    )
    return clf, history


def train_lgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
) -> Tuple[Any, Dict[str, List[float]]]:
    if lgb is None:
        raise RuntimeError("LightGBM bulunamadı; `pip install lightgbm` gerekli.")
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }
    train_dataset = lgb.Dataset(train_x, label=train_y)
    val_dataset = lgb.Dataset(val_x, label=val_y, reference=train_dataset)
    model = lgb.train(
        params,
        train_dataset,
        valid_sets=[val_dataset],
        num_boost_round=500,
        early_stopping_rounds=30,
        verbose_eval=50,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    return model, history


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    class_count = np.bincount(labels, minlength=num_classes)
    total = labels.shape[0]
    weights = total / (num_classes * np.maximum(class_count, 1))
    return weights.astype(np.float32)


def evaluate_classifier(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    classifier_type: str,
    device: torch.device,
) -> np.ndarray:
    if classifier_type == "mlp":
        model.eval()
        tensor = torch.from_numpy(features).float().to(device)
        with torch.no_grad():
            logits = model(tensor)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds
    if classifier_type == "logreg":
        return model.predict(features)
    if classifier_type == "lgbm":
        probs = model.predict(features)
        return np.argmax(probs, axis=1)
    raise ValueError(f"Bilinmeyen sınıflandırıcı: {classifier_type}")


def plot_training_history(history: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    if history["train_loss"]:
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.twinx()
        plt.plot(history["train_acc"], color="green", alpha=0.6, label="Train Acc")
        plt.plot(history["val_acc"], color="red", alpha=0.6, label="Val Acc")
        plt.ylabel("Accuracy")
        plt.title("Training History")
        plt.legend(loc="lower right")
    else:
        plt.plot(history["train_acc"], marker="o", label="Train Acc")
        plt.plot(history["val_acc"], marker="o", label="Val Acc")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Learning Curve")
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path_raw: Path,
    out_path_norm: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path_raw, dpi=160)
    plt.close(fig)

    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path_norm, dpi=160)
    plt.close(fig)


def plot_class_accuracy(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    correct = np.diag(cm)
    total = cm.sum(axis=1).clip(min=1)
    acc = correct / total
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(class_names), y=acc, palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title("Class-wise Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_sample_predictions(
    waveforms: Sequence[np.ndarray],
    true_labels: Sequence[str],
    pred_labels: Sequence[str],
    sr: int,
    out_path: Path,
    num_samples: int = 6,
) -> None:
    count = min(num_samples, len(waveforms))
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.2 * count))
    if count == 1:
        axes = [axes]  # type: ignore
    for ax, waveform, true_label, pred_label in zip(axes, waveforms[:count], true_labels[:count], pred_labels[:count]):
        t = np.linspace(0, len(waveform) / sr, num=len(waveform))
        ax.plot(t, waveform, color="#264653")
        ax.set_title(f"True: {true_label} | Pred: {pred_label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amp")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_model_artifacts(
    out_dir: Path,
    backbone: nn.Module,
    classifier: Any,
    classifier_type: str,
    label_encoder: LabelEncoder,
    config: Dict[str, Any],
) -> None:
    artifact = {
        "backbone_state": backbone.state_dict(),
        "classifier": classifier,
        "classifier_type": classifier_type,
        "classes": list(label_encoder.classes_),
        "config": config,
    }
    joblib.dump(artifact, out_dir / "panns_cnn14_model.pkl")


def save_results_json(
    out_dir: Path,
    model_type: str,
    test_accuracy: float,
    class_names: Sequence[str],
    report: Dict[str, Any],
    config: Dict[str, Any],
    cv_stats: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_type": model_type,
        "classifier": config.get("classifier"),
        "test_loss": None,
        "test_accuracy": float(test_accuracy),
        "class_names": list(class_names),
        "classification_report": report,
        "cv": cv_stats or None,
        "config": config,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def maybe_cache_embedding(
    cache_dir: Optional[Path],
    identifier: str,
    compute_fn,
) -> np.ndarray:
    if cache_dir is None:
        return compute_fn()
    cache_path = cache_dir / f"{identifier}.npy"
    if cache_path.exists():
        print(f"Loaded cached embedding for {identifier}")
        return np.load(cache_path)
    embedding = compute_fn()
    np.save(cache_path, embedding)
    return embedding


def read_folds_metadata(csv_path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("slice_file_name")
            fold = row.get("fold")
            if filename and fold:
                mapping[filename] = int(fold)
    if not mapping:
        raise RuntimeError("folds_csv dosyasında slice_file_name ve fold sütunları bulunamadı.")
    return mapping


def split_data(
    samples: Sequence[AudioSample],
    labels: np.ndarray,
    args: argparse.Namespace,
    fold_mapping: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    if args.use_official_folds:
        if not fold_mapping:
            raise RuntimeError("--use_official_folds için folds_csv gereklidir.")
        folds = sorted(set(fold_mapping.values()))
        fold_runs: List[Dict[str, Any]] = []
        for fold in folds:
            train_idx = []
            test_idx = []
            for idx, sample in enumerate(samples):
                key = sample.path.name
                sample_fold = fold_mapping.get(key)
                if sample_fold == fold:
                    test_idx.append(idx)
                else:
                    train_idx.append(idx)
            train_labels = labels[train_idx]
            train_idx_arr, val_idx_arr = train_test_split(
                train_idx,
                test_size=args.val_split,
                stratify=train_labels,
                random_state=args.seed,
            )
            fold_runs.append(
                {
                    "fold": fold,
                    "train_idx": train_idx_arr,
                    "val_idx": val_idx_arr,
                    "test_idx": test_idx,
                }
            )
        return fold_runs
    train_idx, test_idx = train_test_split(
        np.arange(len(samples)),
        test_size=args.test_split,
        stratify=labels,
        random_state=args.seed,
    )
    train_labels = labels[train_idx]
    train_idx_arr, val_idx_arr = train_test_split(
        train_idx,
        test_size=args.val_split,
        stratify=train_labels,
        random_state=args.seed,
    )
    return [
        {
            "fold": None,
            "train_idx": train_idx_arr,
            "val_idx": val_idx_arr,
            "test_idx": test_idx,
        }
    ]


def compute_embeddings_for_indices(
    backbone: nn.Module,
    samples: Sequence[AudioSample],
    indices: Sequence[int],
    sr: int,
    duration: float,
    batch_size: int,
    device: torch.device,
    augment: str,
    snr_values: Sequence[float],
    spec_aug: Optional[SpecAugment],
    cache_dir: Optional[Path],
) -> Tuple[np.ndarray, List[np.ndarray]]:
    waveforms: List[np.ndarray] = []
    stored_waveforms: List[np.ndarray] = []
    for idx in indices:
        sample = samples[idx]
        waveform = load_waveform(sample.path, sr=sr, duration=duration)
        stored_waveforms.append(waveform.copy())
        augmented = apply_augmentation(waveform, sr, augment, snr_values, spec_aug)
        waveforms.append(augmented)

    def run_model() -> np.ndarray:
        return extract_embeddings(backbone, waveforms, sr, batch_size, device)

    identifier = "batch_" + str(abs(hash(tuple(samples[i].path.as_posix() for i in indices))))
    embeddings = maybe_cache_embedding(cache_dir, identifier, run_model)
    return embeddings, stored_waveforms


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PANNs CNN14 feature extractor + classifier.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset klasörü (class alt klasörleri).")
    parser.add_argument("--sr", type=int, default=32_000, help="Örnekleme oranı.")
    parser.add_argument("--duration", type=float, default=4.0, help="Segment süresi (s).")
    parser.add_argument("--folds_csv", type=str, default=None, help="UrbanSound8K metadata CSV yolu.")
    parser.add_argument("--use_official_folds", action="store_true", help="Resmi 10-fold protokolünü uygula.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test oranı (resmi fold yoksa).")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation oranı.")
    parser.add_argument("--classifier", type=str, default="mlp", choices=["logreg", "mlp", "lgbm"], help="Sınıflandırıcı türü.")
    parser.add_argument("--mlp_hidden", type=str, default="256,128", help="MLP gizli katman genişlikleri.")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding çıkarma batch boyutu.")
    parser.add_argument("--epochs", type=int, default=50, help="MLP eğitim epoch sayısı.")
    parser.add_argument("--lr", type=float, default=1e-3, help="MLP öğrenme oranı.")
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Backbone'u dondur.")
    parser.add_argument("--finetune_last_k", type=int, default=0, help="Son k katmanı fine-tune et.")
    parser.add_argument("--augment", type=str, default="none", choices=["none", "spec", "noise"], help="Augment seçeneği.")
    parser.add_argument("--snr_db", type=str, default="5,0,-5", help="Noise augment SNR değerleri.")
    parser.add_argument("--results_root", type=str, default="results", help="Sonuç kök dizini.")
    parser.add_argument("--cache_embeddings", action="store_true", help="Embeddingleri diske cachele.")
    parser.add_argument("--class_weights", action="store_true", help="Sınıf ağırlıkları kullan.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    samples, class_names = discover_dataset(dataset_path)
    print(f"{len(samples)} dosya bulundu. Sınıflar: {class_names}")

    folds_mapping = None
    if args.use_official_folds:
        if not args.folds_csv:
            raise RuntimeError("--use_official_folds için --folds_csv verilmelidir.")
        folds_mapping = read_folds_metadata(Path(args.folds_csv))

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform([s.label for s in samples])

    results_root = Path(args.results_root)
    out_dir = results_root / "panns_cnn14"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "embeddings_cache" if args.cache_embeddings else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    spec_aug = SpecAugment() if args.augment == "spec" else None
    snr_values = [float(x.strip()) for x in args.snr_db.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone = load_panns_cnn14(sample_rate=args.sr, freeze=args.freeze_backbone)
    finetune_params = get_finetune_layers(backbone, args.finetune_last_k)
    if finetune_params:
        print(f"Fine-tuning last {len(finetune_params)} parameters.")

    runs = split_data(samples, labels_encoded, args, folds_mapping)

    cv_metrics: List[float] = []
    best_run: Optional[Dict[str, Any]] = None
    best_accuracy = -1.0

    for run in runs:
        fold_label = run["fold"]
        print(f"\n=== Fold {fold_label if fold_label is not None else 'N/A'} ===")
        train_idx = run["train_idx"]
        val_idx = run["val_idx"]
        test_idx = run["test_idx"]

        train_embeddings, _ = compute_embeddings_for_indices(
            backbone,
            samples,
            train_idx,
            args.sr,
            args.duration,
            args.batch_size,
            device,
            args.augment,
            snr_values,
            spec_aug,
            cache_dir,
        )
        val_embeddings, _ = compute_embeddings_for_indices(
            backbone,
            samples,
            val_idx,
            args.sr,
            args.duration,
            args.batch_size,
            device,
            "none",
            snr_values,
            spec_aug,
            cache_dir,
        )
        test_embeddings, test_waveforms = compute_embeddings_for_indices(
            backbone,
            samples,
            test_idx,
            args.sr,
            args.duration,
            args.batch_size,
            device,
            "none",
            snr_values,
            spec_aug,
            cache_dir,
        )

        train_labels = labels_encoded[train_idx]
        val_labels = labels_encoded[val_idx]
        test_labels = labels_encoded[test_idx]

        class_weights = None
        if args.class_weights:
            class_weights = compute_class_weights(train_labels, len(class_names))

        hidden_layers = tuple(int(x.strip()) for x in args.mlp_hidden.split(",") if x.strip()) or (256, 128)
        classifier_type = args.classifier
        history: Dict[str, List[float]]

        if classifier_type == "mlp":
            classifier, history = train_mlp(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                class_names,
                hidden_layers,
                args.lr,
                args.epochs,
                batch_size=64,
                device=device,
                class_weights=class_weights,
            )
        elif classifier_type == "logreg":
            cw = "balanced" if args.class_weights else None
            classifier, history = train_logreg(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                class_weight=cw,
            )
        else:
            classifier, history = train_lgbm(
                train_embeddings,
                train_labels,
                val_embeddings,
                val_labels,
                num_classes=len(class_names),
            )

        preds = evaluate_classifier(
            classifier,
            test_embeddings,
            test_labels,
            classifier_type,
            device,
        )

        accuracy = accuracy_score(test_labels, preds)
        print(f"Test accuracy: {accuracy:.4f}")
        cv_metrics.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run = {
                "fold": fold_label,
                "classifier": classifier,
                "history": history,
                "test_preds": preds,
                "test_waveforms": test_waveforms,
                "test_labels": test_labels,
                "train_embeddings": train_embeddings,
                "val_embeddings": val_embeddings,
                "test_embeddings": test_embeddings,
            }

    if best_run is None:
        raise RuntimeError("Eğitim başarısız, en iyi koşu bulunamadı.")

    best_classifier = best_run["classifier"]
    report = classification_report(
        best_run["test_labels"],
        best_run["test_preds"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(best_run["test_labels"], best_run["test_preds"])

    plot_training_history(best_run["history"], out_dir / "training_history.png")
    plot_confusion_matrices(
        cm,
        class_names,
        out_dir / "confusion_matrix.png",
        out_dir / "confusion_matrix_normalized.png",
    )
    plot_class_accuracy(cm, class_names, out_dir / "class_accuracy.png")
    example_waveforms = [best_run["test_waveforms"][i] for i in range(len(best_run["test_waveforms"]))]
    true_labels = [class_names[idx] for idx in best_run["test_labels"]]
    pred_labels = [class_names[idx] for idx in best_run["test_preds"]]
    plot_sample_predictions(
        example_waveforms,
        true_labels,
        pred_labels,
        args.sr,
        out_dir / "sample_predictions.png",
    )

    model_type = f"PANNs CNN14 (frozen) + {args.classifier.upper()}"
    config = vars(args)
    cv_stats = None
    if args.use_official_folds:
        mean_acc = float(np.mean(cv_metrics))
        std_acc = float(np.std(cv_metrics))
        cv_stats = {
            "folds": len(cv_metrics),
            "metric": "accuracy",
            "mean": mean_acc,
            "std": std_acc,
        }
        print(f"CV accuracy mean={mean_acc:.4f} std={std_acc:.4f}")

    save_model_artifacts(out_dir, backbone, best_classifier, args.classifier, label_encoder, config)
    save_results_json(out_dir, model_type, best_accuracy, class_names, report, config, cv_stats)

    print(f"Bitti. Sonuçlar {out_dir} altında.")


if __name__ == "__main__":
    main()
