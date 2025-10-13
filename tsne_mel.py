#!/usr/bin/env python3
"""
Generate a t-SNE visualization using log-mel spectrogram features
computed from a UrbanSound-style dataset (dataset/<class>/*.wav).

The script loads a configurable number of audio files per class, extracts
log-mel spectrogram statistics, applies an optional PCA pre-reduction and
then t-SNE to obtain 2-D embeddings, finally saving the scatter plot under
results/tsne_mel/.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


def discover_dataset(root: Path) -> Dict[str, List[Path]]:
    class_to_files: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        wavs = sorted(class_dir.glob("*.wav"))
        if wavs:
            class_to_files[class_dir.name] = wavs
    if not class_to_files:
        raise RuntimeError(f"Hiç .wav dosyası bulunamadı: {root}")
    return class_to_files


def load_log_mel(
    path: Path,
    sr: int,
    duration: float,
    n_fft: int,
    hop: int,
    n_mels: int,
    fmin: int,
    fmax: int,
) -> np.ndarray:
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def summarise_log_mel(log_mel: np.ndarray, strategy: str = "flatten") -> np.ndarray:
    if strategy == "flatten":
        return log_mel.flatten()
    if strategy == "meanstd":
        mean = log_mel.mean(axis=1)
        std = log_mel.std(axis=1)
        return np.concatenate([mean, std])
    raise ValueError(f"Bilinmeyen özetleme stratejisi: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="t-SNE plot using Mel spectrogram features.")
    parser.add_argument("--dataset", type=str, default="dataset", help="dataset/<class>/*.wav kökü")
    parser.add_argument("--sr", type=int, default=32_000, help="Örnekleme oranı")
    parser.add_argument("--duration", type=float, default=4.0, help="Segment süresi (s)")
    parser.add_argument("--n_fft", type=int, default=1_024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--fmin", type=int, default=20)
    parser.add_argument("--fmax", type=int, default=8_000)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--pca_components", type=int, default=50, help="t-SNE öncesi PCA bileşen sayısı (0 kapatır)")
    parser.add_argument("--limit_per_class", type=int, default=150, help="Her sınıftan max dosya sayısı (0: sınırsız)")
    parser.add_argument("--summary", type=str, default="meanstd", choices=["flatten", "meanstd"], help="Mel özetleme yöntemi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_root", type=str, default="results", help="Çıktı kök klasörü")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    classes = discover_dataset(dataset_root)

    features: List[np.ndarray] = []
    labels: List[str] = []

    for class_name, files in classes.items():
        max_items = args.limit_per_class if args.limit_per_class > 0 else len(files)
        selected = files[:max_items]
        for wav_path in tqdm(selected, desc=f"{class_name:>20}", leave=False):
            log_mel = load_log_mel(
                wav_path,
                sr=args.sr,
                duration=args.duration,
                n_fft=args.n_fft,
                hop=args.hop,
                n_mels=args.n_mels,
                fmin=args.fmin,
                fmax=args.fmax,
            )
            feature = summarise_log_mel(log_mel, args.summary)
            features.append(feature)
            labels.append(class_name)

    feature_matrix = np.stack(features)
    print(f"Özellik matris boyutu: {feature_matrix.shape}")

    if args.pca_components > 0 and feature_matrix.shape[1] > args.pca_components:
        print(f"PCA ile {args.pca_components} boyuta indirgeniyor...")
        pca = PCA(n_components=args.pca_components, random_state=args.seed)
        feature_matrix = pca.fit_transform(feature_matrix)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="random",
        random_state=args.seed,
    )
    embeddings = tsne.fit_transform(feature_matrix)

    results_dir = Path(args.results_root) / "tsne_mel"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=labels,
        palette="tab10",
        s=30,
        alpha=0.8,
    )
    plt.title("Mel Spektrogram Tabanlı t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best", fontsize="small", markerscale=1, frameon=False, ncol=2)
    plt.tight_layout()
    out_path = results_dir / "tsne_mel.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    np.save(results_dir / "tsne_embeddings.npy", embeddings)
    np.save(results_dir / "tsne_labels.npy", np.array(labels))
    print(f"t-SNE grafiği kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
