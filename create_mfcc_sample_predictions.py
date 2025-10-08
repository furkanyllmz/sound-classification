import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import json

# Sonuçları yükle
with open('results/mfcc/results.json', 'r') as f:
    results = json.load(f)

class_names = results['class_names']

# Model ve veriyi yükle (sadece görselleştirme için gerekli veriyi yeniden oluşturuyoruz)
# Veri setini tekrar yükleyip test split yapacağız
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# MFCC özellikleri çıkar
def extract_mfcc(file_path, max_pad_len=174):
    """MFCC özelliklerini ses dosyasından çıkarır"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        print(f"Hata {file_path}: {e}")
        return None

# Veri setini yükle
def load_dataset(dataset_path='dataset'):
    """Tüm ses dosyalarını yükler ve MFCC özelliklerini çıkarır"""
    features = []
    labels = []

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()

    print(f"Veri yükleniyor...")

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        count = 0

        for filename in os.listdir(class_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_path, filename)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
                    count += 1

        print(f"{label}: {count} dosya yüklendi")

    return np.array(features), np.array(labels), classes

print("Veri seti yükleniyor...")
X, y, _ = load_dataset()

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Veriyi reshape et
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Train-test split (aynı random_state ile)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

print(f"\nTest set: {X_test.shape}")

# Modeli yükle
print("\nModel yükleniyor...")
model = load_model('results/mfcc/mfcc_cnn_model.h5')

# Tahminler
print("Tahminler yapılıyor...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Örnek MFCC görselleştirmesi
print("\nÖrnek MFCC görselleştirmeleri oluşturuluyor...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(min(6, len(X_test))):
    axes[i].imshow(X_test[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    true_label = class_names[y_true_classes[i]]
    pred_label = class_names[y_pred_classes[i]]

    # Doğru/yanlış tahmine göre renk değiştir
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}',
                     fontweight='bold', color=color, fontsize=11)
    axes[i].set_xlabel('Time Frames', fontsize=9)
    axes[i].set_ylabel('MFCC Coefficients', fontsize=9)

plt.suptitle('MFCC-CNN: Sample Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mfcc/sample_predictions.png', dpi=300, bbox_inches='tight')
print("✅ Örnek tahminler kaydedildi: results/mfcc/sample_predictions.png")
plt.close()

print("\n" + "="*60)
print("Sample predictions başarıyla oluşturuldu!")
print("="*60)
