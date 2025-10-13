#!/usr/bin/env python3
"""
Produce all pre-model methodology assets (figures, markdown, captions, report)
for the sound classification project — polished visuals + dataset EDA + t-SNE/UMAP.

Outputs under results/methodology/.
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from sklearn.manifold import TSNE

# Try UMAP (optional)
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("dataset")
OUT_DIR = Path("results/methodology")

SR = 16_000
SEG_SECONDS = 3.0

NFFT = 1024
HOP = 256
WIN = 1024
MELS = 128
FMIN = 20
FMAX = 8_000

AUG_SNR_LEVELS = [5, 0, -5, -10]
TIME_STRETCH = [0.9, 1.1]
PITCH_SHIFT = [-2, 2]

AUG_PROBABILITIES = {"snr": 0.6, "time_stretch": 0.4, "pitch_shift": 0.4}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RNG = random.Random(2025)
NP_RNG = np.random.default_rng(2025)

# Matplotlib defaults (clean, readable, paper-friendly)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": True,
    }
)

CB_PALETTE = {
    "train": "#3B7EA1",  # blue
    "val": "#E6A700",    # amber
    "test": "#D1495B",   # red
    "base": "#264653",
    "accent": "#2A9D8F",
    "accent2": "#8AB17D",
    "warn": "#E76F51",
    "muted": "#6C757D",
}
SPEC_CMAP = "magma"


# ---------------------------------------------------------------------------
# Data structures & helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioFile:
    path: Path
    class_name: str
    track_id: str


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_dataset(data_dir: Path) -> List[AudioFile]:
    audio_files: List[AudioFile] = []
    if not data_dir.exists():
        raise RuntimeError(f"Veri klasörü bulunamadı: {data_dir}")

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        wavs = sorted(list(class_dir.glob("*.wav")))
        for wav_path in wavs:
            # Track id: "recordingId-anything" -> first two stem parts if exist
            name_parts = wav_path.stem.split("-")
            track_id = "-".join(name_parts[:2]) if len(name_parts) >= 2 else wav_path.stem
            audio_files.append(AudioFile(wav_path, class_dir.name, track_id))

    if not audio_files:
        raise RuntimeError(f"Veri klasöründe .wav dosyası bulunamadı: {data_dir}")
    return audio_files


def compute_track_splits(
    audio_files: Sequence[AudioFile],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Dict[str, str]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    tracks = sorted({f.track_id for f in audio_files})
    rng.shuffle(tracks)
    total = len(tracks)

    train_cut = int(round(total * train_ratio))
    val_cut = train_cut + int(round(total * val_ratio))

    track_splits: Dict[str, str] = {}
    for idx, track in enumerate(tracks):
        if idx < train_cut:
            split = "train"
        elif idx < val_cut:
            split = "val"
        else:
            split = "test"
        track_splits[track] = split
    return track_splits


def aggregate_class_counts(items: Iterable[AudioFile]) -> Counter:
    counter: Counter = Counter()
    for it in items:
        counter[it.class_name] += 1
    return counter


def load_segment(
    path: Path, target_sr: int = SR, segment_seconds: float = SEG_SECONDS
) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    seg_samples = int(segment_seconds * target_sr)
    if len(audio) < seg_samples:
        pad = seg_samples - len(audio)
        audio = np.pad(audio, (0, pad))
    else:
        # center crop
        start = max(0, (len(audio) - seg_samples) // 2)
        audio = audio[start : start + seg_samples]
    # z-score normalize (robust)
    astd = np.std(audio) + 1e-8
    audio = (audio - np.mean(audio)) / astd
    return audio.astype(np.float32), target_sr


def calculate_augmentation_multiplier() -> float:
    return (
        1.0
        + len(AUG_SNR_LEVELS) * AUG_PROBABILITIES["snr"]
        + len(TIME_STRETCH) * AUG_PROBABILITIES["time_stretch"]
        + len(PITCH_SHIFT) * AUG_PROBABILITIES["pitch_shift"]
    )


def _match_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        return y[:target_len]
    pad = target_len - len(y)
    return np.pad(y, (0, pad))


def _mix_with_noise(y: np.ndarray, target_snr_db: float) -> np.ndarray:
    rms_signal = np.sqrt(np.mean(y**2) + 1e-12)
    snr_linear = 10 ** (target_snr_db / 20)
    noise_rms = rms_signal / snr_linear
    noise = NP_RNG.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    current_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    scaled_noise = noise * (noise_rms / current_rms)
    return y + scaled_noise


# ---------------------------------------------------------------------------
# EDA helpers (durations, summary)
# ---------------------------------------------------------------------------

def compute_duration_seconds(path: Path, sr: int = SR) -> float:
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return float(len(y)) / sr
    except Exception:
        return 0.0


def build_feature_vector(y: np.ndarray, sr: int) -> np.ndarray:
    """Compact per-clip vector for TSNE/UMAP: MFCC stats + Chroma mean + spectral stats."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=NFFT, hop_length=HOP, win_length=WIN,
        n_mels=MELS, fmin=FMIN, fmax=FMAX, power=2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)  # (20, T)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, T)
    chroma_mean = chroma.mean(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=NFFT, hop_length=HOP).mean()
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=NFFT, hop_length=HOP).mean()
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=NFFT, hop_length=HOP).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()

    vec = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, [centroid, bw, roll, zcr]])
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure creators (polished)
# ---------------------------------------------------------------------------

def create_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.axis("off")
    steps = [
        "Ham ses\n(dataset)",
        f"Segmentasyon\n({SEG_SECONDS:.1f} sn @ {SR} Hz)",
        "Sessizlik kırpma\n+ Z-score",
        "Öznitelik çıkarımı\n(STFT, Mel, MFCC...)",
        "Parça-temelli bölme\n(80/10/10)",
    ]
    x_positions = np.linspace(0.06, 0.82, len(steps))
    y = 0.5
    w, h = 0.16, 0.36
    for i, (step, x) in enumerate(zip(steps, x_positions)):
        rect = patches.FancyBboxPatch(
            (x, y - h / 2), w, h, boxstyle="round,pad=0.02",
            linewidth=2, edgecolor=CB_PALETTE["base"], facecolor=CB_PALETTE["accent"], alpha=0.95
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y, step, ha="center", va="center", color="white", fontweight="bold")
        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(x + w, y), xytext=(x_positions[i + 1], y),
                arrowprops=dict(arrowstyle="->", linewidth=2.2, color=CB_PALETTE["base"], shrinkA=10, shrinkB=12),
            )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_no_leakage_diagram(
    audio_files: Sequence[AudioFile],
    track_splits: Dict[str, str],
    out_path: Path,
    rng: random.Random,
) -> None:
    track_to_segments: Dict[str, List[AudioFile]] = defaultdict(list)
    for it in audio_files:
        track_to_segments[it.track_id].append(it)

    tracks = list(track_to_segments.keys())
    rng.shuffle(tracks)
    sampled_tracks = tracks[:8] if len(tracks) >= 8 else tracks

    split_positions = {"train": 0.16, "val": 0.5, "test": 0.84}
    colors = {"train": CB_PALETTE["train"], "val": CB_PALETTE["val"], "test": CB_PALETTE["test"]}

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axis("off")

    for label, x in split_positions.items():
        ax.text(x, len(sampled_tracks) + 0.6, label.upper(), ha="center", va="bottom", fontweight="bold")
        ax.add_line(plt.Line2D((x - 0.18, x - 0.18), (-0.4, len(sampled_tracks) - 0.2),
                               color=CB_PALETTE["muted"], linewidth=1, alpha=0.35))

    for row, track_id in enumerate(sampled_tracks):
        segments = sorted(track_to_segments[track_id], key=lambda s: s.path.name)
        split = track_splits.get(track_id, "train")
        base_x = split_positions[split]
        ax.text(0.01, row, track_id, ha="left", va="center", fontweight="bold", fontsize=9)
        for idx, seg in enumerate(segments[:10]):  # cap to avoid overflow
            x = base_x - 0.18 + idx * 0.08
            rect = patches.FancyBboxPatch(
                (x, row - 0.35), 0.14, 0.6, boxstyle="round,pad=0.02",
                linewidth=1.5, edgecolor=colors[split], facecolor=colors[split], alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(x + 0.07, row - 0.05, seg.path.stem, ha="center", va="center",
                    fontsize=7, rotation=90, color="black")

    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_class_distribution_plot(
    class_counts: Counter, augmented_counts: Dict[str, int], out_path: Path
) -> None:
    classes = sorted(class_counts.keys())
    baseline = np.array([class_counts[c] for c in classes], dtype=float)
    augmented = np.array([augmented_counts[c] for c in classes], dtype=float)

    x = np.arange(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - width / 2, baseline, width=width, color=CB_PALETTE["base"], label="Ham veri")
    ax.bar(x + width / 2, augmented, width=width, color=CB_PALETTE["accent"], label="Aug sonrası")
    ax.set_ylabel("Örnek sayısı")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=35, ha="right")
    for i, v in enumerate(baseline):
        ax.text(i - width / 2, v + max(1, 0.01 * v), f"{int(v)}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(augmented):
        ax.text(i + width / 2, v + max(1, 0.01 * v), f"{int(v)}", ha="center", va="bottom", fontsize=9, color=CB_PALETTE["muted"])
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_stft_windowing_diagram(out_path: Path) -> None:
    duration_s = 0.2
    t = np.linspace(0, duration_s, int(SR * duration_s))
    signal = np.sin(2 * np.pi * 220 * t) + 0.4 * np.sin(2 * np.pi * 440 * t)

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.plot(t, signal, color=CB_PALETTE["base"], linewidth=1.8)
    ax.set_xlabel("Zaman (s)"); ax.set_ylabel("Genlik")

    hop_time = HOP / SR
    win_time = WIN / SR
    min_y, max_y = signal.min() - 0.3, signal.max() + 0.3

    for idx in range(4):
        start = idx * hop_time
        rect = patches.Rectangle(
            (start, min_y), win_time, max_y - min_y, linewidth=1.4,
            edgecolor=CB_PALETTE["val"], facecolor=CB_PALETTE["val"], alpha=0.25,
        )
        ax.add_patch(rect)
        ax.text(start + win_time / 2, max_y + 0.05, f"Pencere {idx + 1}",
                ha="center", va="bottom", color=CB_PALETTE["warn"])

    param_text = (
        f"n_fft: {NFFT}\n"
        f"win_length: {WIN} örnek (~{win_time*1000:.1f} ms)\n"
        f"hop_length: {HOP} örnek (~{hop_time*1000:.1f} ms)"
    )
    ax.text(0.65, 0.15, param_text, transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor=CB_PALETTE["base"], boxstyle="round,pad=0.4"))
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_feature_grid(audio: np.ndarray, sr: int, out_path: Path) -> None:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=NFFT, hop_length=HOP, win_length=WIN,
        n_mels=MELS, fmin=FMIN, fmax=FMAX, power=2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    harmonic, percussive = librosa.effects.hpss(audio)
    stft_h = librosa.stft(harmonic, n_fft=NFFT, hop_length=HOP, win_length=WIN)
    stft_p = librosa.stft(percussive, n_fft=NFFT, hop_length=HOP, win_length=WIN)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    tempogram = librosa.feature.tempogram(y=audio, sr=sr, hop_length=HOP)

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
    for ax in axes.flat: ax.grid(False)

    librosa.display.specshow(mel_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                             fmin=FMIN, fmax=FMAX, ax=axes[0, 0], cmap=SPEC_CMAP)
    axes[0, 0].set_title("Mel spektrogram")

    librosa.display.specshow(mfcc, x_axis="time", ax=axes[0, 1], cmap=SPEC_CMAP)
    axes[0, 1].set_title("MFCC (20)")
    axes[0, 1].set_ylabel("Katsayı")

    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", cmap=SPEC_CMAP, ax=axes[0, 2])
    axes[0, 2].set_title("Chroma")

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_h), ref=np.max),
                             sr=sr, hop_length=HOP, x_axis="time", y_axis="hz", cmap=SPEC_CMAP, ax=axes[1, 0])
    axes[1, 0].set_title("Harmonik")

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_p), ref=np.max),
                             sr=sr, hop_length=HOP, x_axis="time", y_axis="hz", cmap=SPEC_CMAP, ax=axes[1, 1])
    axes[1, 1].set_title("Perküsyel")

    librosa.display.specshow(tempogram, sr=sr, hop_length=HOP, x_axis="time", y_axis="tempo",
                             cmap=SPEC_CMAP, ax=axes[1, 2])
    axes[1, 2].set_title("Tempogram")

    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_augmentation_effects_grid(audio: np.ndarray, sr: int, out_path: Path) -> None:
    variants: List[Tuple[str, np.ndarray]] = [("Orijinal", audio)]
    for snr in AUG_SNR_LEVELS:
        variants.append((f"SNR {snr:+} dB", _mix_with_noise(audio, snr)))
    for rate in TIME_STRETCH:
        stretched = _match_length(librosa.effects.time_stretch(audio, rate=rate), len(audio))
        variants.append((f"Tempo x{rate:.1f}", stretched))
    for steps in PITCH_SHIFT:
        shifted = _match_length(librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps), len(audio))
        variants.append((f"Pitch {steps:+}", shifted))

    cols = 4
    rows = math.ceil(len(variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 6.4))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat: ax.axis("off")

    for ax, (title, yv) in zip(axes.flat, variants):
        mel = librosa.feature.melspectrogram(
            y=yv, sr=sr, n_fft=NFFT, hop_length=HOP, win_length=WIN,
            n_mels=MELS, fmin=FMIN, fmax=FMAX, power=2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=HOP, x_axis="time", y_axis=None, cmap=SPEC_CMAP, ax=ax)
        ax.set_title(title, fontsize=11)

    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def create_snr_protocol_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.axis("off")
    blocks = [
        ("Temiz segment\nx_clean", (0.08, 0.5), CB_PALETTE["accent"]),
        ("Arka plan gürültüsü\nn_bg", (0.38, 0.5), CB_PALETTE["val"]),
        (r"Ölçek α = rms(x)/10^(SNR/20)", (0.66, 0.7), "#f4a261"),
        ("Karışım\nx_mix = x_clean + α·n_bg", (0.66, 0.3), CB_PALETTE["warn"]),
    ]
    for text, (x, y), color in blocks:
        rect = patches.FancyBboxPatch((x - 0.12, y - 0.18), 0.24, 0.32, boxstyle="round,pad=0.03",
                                      linewidth=1.6, edgecolor=CB_PALETTE["base"], facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", color="black", fontweight="bold")
    arrow = dict(arrowstyle="->", linewidth=2, color=CB_PALETTE["base"], shrinkA=12, shrinkB=12)
    ax.annotate("", xy=(0.26, 0.5), xytext=(0.2, 0.5), arrowprops=arrow)
    ax.annotate("", xy=(0.54, 0.5), xytext=(0.46, 0.5), arrowprops=arrow)
    ax.annotate("", xy=(0.66, 0.55), xytext=(0.66, 0.48), arrowprops=arrow)
    ax.annotate("", xy=(0.66, 0.42), xytext=(0.66, 0.35), arrowprops=arrow)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# ----------------------- t-SNE / UMAP -----------------------

def create_embeddings_plots(audio_files: Sequence[AudioFile], out_tsne: Path, out_umap: Path | None) -> Tuple[int, List[str]]:
    """
    Build per-clip compact vectors and draw TSNE (and UMAP if available).
    Returns: (n_samples, class_list)
    """
    vectors = []
    labels = []
    classes_set = set()

    # cap total for speed if dataset is huge (keeps class balance)
    per_class_cap = 300

    # group by class
    by_class: Dict[str, List[AudioFile]] = defaultdict(list)
    for af in audio_files:
        by_class[af.class_name].append(af)

    for cls, items in by_class.items():
        take = items[:per_class_cap]
        for it in take:
            y, _ = load_segment(it.path)
            vec = build_feature_vector(y, SR)
            vectors.append(vec)
            labels.append(cls)
        classes_set.add(cls)

    X = np.vstack(vectors).astype(np.float32)
    y = np.array(labels)
    class_list = sorted(list(classes_set))

    # --- TSNE ---
    perplexity = max(5, min(30, (len(X) // 3)))  # safe bounds
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="random", random_state=2025)
    Z = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    colors = plt.cm.get_cmap("tab10", len(class_list))
    for i, cls in enumerate(class_list):
        mask = (y == cls)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=14, alpha=0.8, label=cls, marker="o")
    ax.set_title("t-SNE (MFCC/Chroma + spektral özet) — 2B gömme")
    ax.set_xlabel("Bileşen 1"); ax.set_ylabel("Bileşen 2")
    ax.legend(ncol=2, fontsize=9, markerscale=1.1, frameon=False)
    fig.savefig(out_tsne, bbox_inches="tight"); plt.close(fig)

    # --- UMAP (optional) ---
    if HAS_UMAP and out_umap is not None:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=2025)
        U = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(8.8, 6.8))
        for i, cls in enumerate(class_list):
            mask = (y == cls)
            ax.scatter(U[mask, 0], U[mask, 1], s=14, alpha=0.85, label=cls, marker=".")
        ax.set_title("UMAP — 2B gömme")
        ax.set_xlabel("Bileşen 1"); ax.set_ylabel("Bileşen 2")
        ax.legend(ncol=2, fontsize=9, markerscale=1.1, frameon=False)
        fig.savefig(out_umap, bbox_inches="tight"); plt.close(fig)

    return len(X), class_list


# ---------------------------------------------------------------------------
# Text artefacts
# ---------------------------------------------------------------------------

def write_hparams_table(out_path: Path) -> None:
    lines = [
        "| Parametre | Değer |",
        "|-----------|-------|",
        f"| Örnekleme oranı | {SR} Hz |",
        f"| Segment süresi | {SEG_SECONDS:.1f} s |",
        f"| Normalizasyon | Kanal başına Z-score |",
        f"| n_fft | {NFFT} |",
        f"| hop_length | {HOP} |",
        f"| win_length | {WIN} |",
        f"| Mel bant sayısı | {MELS} |",
        f"| fmin | {FMIN} Hz |",
        f"| fmax | {FMAX} Hz |",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preprocessing_md(out_path: Path, class_names: Sequence[str]) -> None:
    class_list = ", ".join(class_names)
    content = f"""# Ön-İşleme Metodolojisi

## Veri Kaynağı ve Segmentasyon
Kayıtlar {SR} Hz örnekleme ile yüklendi; sessiz kısımlar kırpılarak {SEG_SECONDS:.1f} saniyelik segmentlere ayrıldı ve kanal başına Z-score ile normalize edildi.

## Parça-Temelli Bölme
Track kimliği baz alınarak segmentler birden fazla split'e düşmeyecek şekilde %80 eğitim, %10 doğrulama ve %10 test oranlarında ayrıldı.

## Öznitelik Çıkarımı
n_fft={NFFT}, hop_length={HOP}, win_length={WIN} parametreleriyle STFT uygulandı; Mel (n_mels={MELS}, fmin={FMIN} Hz, fmax={FMAX} Hz) spektrogramlarının yanı sıra MFCC, harmonic-percussive ayrımı, Chroma ve Tempogram temsilleri hesaplandı.

## Augmentasyon Protokolü (Plan)
SNR hedefleri {", ".join(f"{snr:+} dB" for snr in AUG_SNR_LEVELS)}; time-stretch oranları {TIME_STRETCH[0]}–{TIME_STRETCH[1]}; pitch-shift adımları {PITCH_SHIFT[0]}–{PITCH_SHIFT[1]} yarım ses olarak planlandı.

## Kalite/Kontrol
Sınıf dağılımı (sınıflar: {class_list}) izlenerek dengesizlikte model dışı stratejiler (sınıf ağırlıklandırma / örnekleme) değerlendirilir.
"""
    out_path.write_text(content, encoding="utf-8")


def write_captions_json(out_path: Path) -> None:
    captions = {
        "01_pipeline.png": "Şekil 1. Uçtan uca veri işleme hattı: segmentasyon, normalizasyon, öznitelik ve parça-temelli bölme.",
        "02_no_leakage.png": "Şekil 2. Parça tabanlı ayrım: aynı kaynaktan segmentler birden fazla split'e dağılmaz.",
        "03_class_distribution.png": "Şekil 3. Sınıf bazında ham ve augmentasyon sonrası örnek sayıları.",
        "04_stft_windowing.png": "Şekil 4. STFT pencere/atlanım (win/hop) ile zaman-frekans çözünürlüğü.",
        "05_feature_grid.png": "Şekil 5. Aynı segmentin farklı temsilleri: Mel, MFCC, Chroma, Harmonik, Perküsyel, Tempogram.",
        "06_augmentation_effects.png": "Şekil 6. Augmentasyonların mel spektrograma etkisi (SNR, tempo, pitch).",
        "07_snr_protocol.png": "Şekil 7. Hedef SNR için α ölçeklemesi ile gürültü karışım protokolü.",
        "09_tsne.png": "Şekil 8. t-SNE ile 2B gömme (MFCC/Chroma + spektral özet).",
        "10_umap.png": "Şekil 9. UMAP ile 2B gömme (varsa).",
    }
    out_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(out_path: Path, outputs: List[Tuple[Path, str]], assumptions: List[str], notes: List[str]) -> None:
    lines = ["# Üretilen Dosyalar"]
    for rel_path, purpose in outputs:
        lines.append(f"- {rel_path.as_posix()} — {purpose}")
    lines.append("")
    lines.append("# Varsayımlar")
    if assumptions:
        for a in assumptions:
            lines.append(f"- {a}")
    else:
        lines.append("- Varsayım bulunmuyor.")
    lines.append("")
    lines.append("# Notlar")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_out_dir()
    notes: List[str] = []
    try:
        audio_files = collect_dataset(DATA_DIR)
    except Exception as e:
        raise SystemExit(f"Hata: {e}")

    class_counts = aggregate_class_counts(audio_files)
    class_names = sorted(class_counts.keys())

    track_splits = compute_track_splits(audio_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RNG)

    # estimated augmented counts (display only)
    augmentation_multiplier = calculate_augmentation_multiplier()
    augmented_counts = {cls: int(round(cnt * augmentation_multiplier)) for cls, cnt in class_counts.items()}

    # A representative segment for grids
    segment_audio, _ = load_segment(audio_files[0].path)

    outputs: List[Tuple[Path, str]] = []

    # Figures
    p01 = OUT_DIR / "01_pipeline.png"
    create_pipeline_diagram(p01)
    outputs.append((p01.relative_to(OUT_DIR.parent), "Veri işleme hattı."))

    p02 = OUT_DIR / "02_no_leakage.png"
    create_no_leakage_diagram(audio_files, track_splits, p02, RNG)
    outputs.append((p02.relative_to(OUT_DIR.parent), "Sızıntı önleme şeması."))

    p03 = OUT_DIR / "03_class_distribution.png"
    create_class_distribution_plot(class_counts, augmented_counts, p03)
    outputs.append((p03.relative_to(OUT_DIR.parent), "Sınıf dağılımı çubuk grafiği."))

    p04 = OUT_DIR / "04_stft_windowing.png"
    create_stft_windowing_diagram(p04)
    outputs.append((p04.relative_to(OUT_DIR.parent), "Pencereleme açıklaması."))

    p05 = OUT_DIR / "05_feature_grid.png"
    create_feature_grid(segment_audio, SR, p05)
    outputs.append((p05.relative_to(OUT_DIR.parent), "Öznitelik ızgarası (Mel/MFCC/Chroma/HP/Tempogram)."))

    p06 = OUT_DIR / "06_augmentation_effects.png"
    create_augmentation_effects_grid(segment_audio, SR, p06)
    outputs.append((p06.relative_to(OUT_DIR.parent), "Augmentasyon etkileri minigrid."))

    p07 = OUT_DIR / "07_snr_protocol.png"
    create_snr_protocol_diagram(p07)
    outputs.append((p07.relative_to(OUT_DIR.parent), "SNR karışım protokolü."))

    # Dataset EDA extras: durations histogram (nice to have)
    durations = np.array([compute_duration_seconds(af.path) for af in audio_files])
    if durations.size > 0:
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        ax.hist(durations, bins=30, color=CB_PALETTE["accent2"])
        ax.set_title("Kayıt süreleri dağılımı")
        ax.set_xlabel("Süre (s)"); ax.set_ylabel("Adet")
        p08 = OUT_DIR / "08_duration_hist.png"
        fig.savefig(p08, bbox_inches="tight"); plt.close(fig)
        outputs.append((p08.relative_to(OUT_DIR.parent), "Kayıt süreleri histogramı."))

    # Embeddings: t-SNE (+ UMAP if available)
    p09 = OUT_DIR / "09_tsne.png"
    p10 = OUT_DIR / "10_umap.png" if HAS_UMAP else None
    n_used, class_list = create_embeddings_plots(audio_files, p09, p10)
    outputs.append((p09.relative_to(OUT_DIR.parent), f"t-SNE 2B gömme (n={n_used})."))
    if p10 is not None and p10.exists():
        outputs.append((p10.relative_to(OUT_DIR.parent), f"UMAP 2B gömme (n={n_used})."))

    # Text artefacts
    t01 = OUT_DIR / "08_hparams_table.md"
    write_hparams_table(t01)
    outputs.append((t01.relative_to(OUT_DIR.parent), "Ön-işleme hiperparametre tablosu."))

    t02 = OUT_DIR / "03_preprocessing.md"
    write_preprocessing_md(t02, class_list if class_list else class_names)
    outputs.append((t02.relative_to(OUT_DIR.parent), "Makale metodoloji (model öncesi) metni."))

    t03 = OUT_DIR / "captions.json"
    write_captions_json(t03)
    outputs.append((t03.relative_to(OUT_DIR.parent), "Türkçe şekil alt yazıları."))

    assumptions = [
        "SNR karışımı beyaz gürültü ile simüle edildi (arka plan gürültüsü sağlanmadıysa).",
        "t-SNE vektörü MFCC(20) ort/ss + Chroma(12) ort + spektral özetlerle kuruldu.",
        f"t-SNE perplexity veri boyutuna göre {max(5, min(30, (n_used // 3)))} seçildi.",
    ]
    if not HAS_UMAP:
        notes.append("UMAP bulunamadı; sadece t-SNE üretildi. (pip install umap-learn ile ekleyebilirsin.)")

    rep = OUT_DIR / "report.txt"
    write_report(rep, outputs, assumptions, notes)
    outputs.append((rep.relative_to(OUT_DIR.parent), "Çıktı özeti ve notlar."))


if __name__ == "__main__":
    main()
