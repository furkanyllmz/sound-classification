import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Sonuçları kaydetmek için klasör
os.makedirs('results/feature_visualizations', exist_ok=True)

# Her sınıftan bir örnek ses dosyası seç
dataset_path = 'dataset'
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
classes.sort()

print(f"Sınıflar: {classes}")
print("Görselleştirmeler oluşturuluyor...\n")

# Her sınıf için MFCC ve Mel-Spectrogram çiz
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)

    # İlk wav dosyasını al
    wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
    if not wav_files:
        print(f"⚠️  {class_name} için ses dosyası bulunamadı")
        continue

    sample_file = os.path.join(class_path, wav_files[0])

    # Ses dosyasını yükle
    audio, sr = librosa.load(sample_file, res_type='kaiser_fast', duration=4.0)

    # MFCC hesapla
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Mel-Spectrogram hesapla
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Görselleştirme
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # MFCC
    img1 = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axes[0], cmap='viridis')
    axes[0].set_title(f'{class_name} - MFCC Features', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('MFCC Coefficients', fontsize=12)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f')

    # Mel-Spectrogram
    img2 = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                     sr=sr, ax=axes[1], cmap='magma')
    axes[1].set_title(f'{class_name} - Mel-Spectrogram', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()

    # Kaydet
    output_file = f'results/feature_visualizations/{class_name}_features.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ {class_name}: {output_file}")
    plt.close()

print("\n" + "="*60)
print("Tüm görselleştirmeler tamamlandı!")
print("Klasör: results/feature_visualizations/")
print("="*60)
