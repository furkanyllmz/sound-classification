import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

# Sonuçları kaydetmek için klasör oluştur
os.makedirs('results/melspectrogram', exist_ok=True)

# Veri setini yükle ve Mel-Spectrogram özellikleri çıkar
def extract_melspectrogram(file_path, n_mels=128, max_len=174):
    """Mel-Spectrogram özelliklerini ses dosyasından çıkarır"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Padding veya truncate
        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_len]

        return mel_spec_db
    except Exception as e:
        print(f"Hata {file_path}: {e}")
        return None

# Veri setini hazırla
def load_dataset(dataset_path='dataset'):
    """Tüm ses dosyalarını yükler ve Mel-Spectrogram özelliklerini çıkarır"""
    features = []
    labels = []

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()

    print(f"Bulunan sınıflar: {classes}")

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        print(f"\n{label} sınıfı işleniyor...")
        count = 0

        for filename in os.listdir(class_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_path, filename)
                mel_spec = extract_melspectrogram(file_path)
                if mel_spec is not None:
                    features.append(mel_spec)
                    labels.append(label)
                    count += 1

                    if count % 100 == 0:
                        print(f"  {count} dosya işlendi...")

        print(f"{label}: Toplam {count} dosya yüklendi")

    return np.array(features), np.array(labels), classes

print("Veri seti yükleniyor...")
X, y, class_names = load_dataset()

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Veriyi reshape et (CNN için)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

print(f"\nVeri seti şekli: {X.shape}")
print(f"Etiket şekli: {y_categorical.shape}")
print(f"Sınıflar: {class_names}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# CNN Modeli Oluştur
def create_melspectrogram_cnn_model(input_shape, num_classes):
    """Mel-Spectrogram tabanlı CNN modeli"""
    model = Sequential([
        # İlk Konvolüsyon Bloğu
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # İkinci Konvolüsyon Bloğu
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Üçüncü Konvolüsyon Bloğu
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dördüncü Konvolüsyon Bloğu
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Fully Connected Katmanlar
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# Modeli oluştur
model = create_melspectrogram_cnn_model(X_train.shape[1:], len(class_names))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Yapısı:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Modeli eğit
print("\nModel eğitimi başlıyor...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Modeli kaydet
model.save('results/melspectrogram/melspectrogram_cnn_model.h5')
print("\nModel kaydedildi: results/melspectrogram/melspectrogram_cnn_model.h5")

# Test verisi üzerinde değerlendirme
print("\nTest verisi üzerinde değerlendirme...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Tahminler
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Sonuçları kaydet
results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'class_names': class_names,
    'classification_report': classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
}

with open('results/melspectrogram/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# ============ GÖRSELLEŞTIRMELER ============

# 1. Training & Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Mel-Spectrogram CNN: Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Mel-Spectrogram CNN: Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/melspectrogram/training_history.png', dpi=300, bbox_inches='tight')
print("Training history grafiği kaydedildi: results/melspectrogram/training_history.png")
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Mel-Spectrogram CNN: Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/melspectrogram/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix kaydedildi: results/melspectrogram/confusion_matrix.png")
plt.close()

# 4. Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Mel-Spectrogram CNN: Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/melspectrogram/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("Normalized confusion matrix kaydedildi: results/melspectrogram/confusion_matrix_normalized.png")
plt.close()

# 5. Class-wise Accuracy Bar Plot
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracy, color='seagreen', edgecolor='black')
plt.title('Mel-Spectrogram CNN: Class-wise Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# Bar'ların üzerine değerleri yaz
for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/melspectrogram/class_accuracy.png', dpi=300, bbox_inches='tight')
print("Class accuracy grafiği kaydedildi: results/melspectrogram/class_accuracy.png")
plt.close()

# 6. Learning Rate Schedule (eğer reduce_lr kullanıldıysa)
if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'])
    plt.title('Mel-Spectrogram CNN: Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/melspectrogram/learning_rate.png', dpi=300, bbox_inches='tight')
    print("Learning rate grafiği kaydedildi: results/melspectrogram/learning_rate.png")
    plt.close()

# 7. Örnek Mel-Spectrogram görselleştirmesi
print("\nÖrnek Mel-Spectrogram görselleştirmeleri oluşturuluyor...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(min(6, len(X_test))):
    axes[i].imshow(X_test[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f'True: {class_names[y_true_classes[i]]}\nPred: {class_names[y_pred_classes[i]]}')
    axes[i].axis('off')

plt.suptitle('Mel-Spectrogram CNN: Sample Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/melspectrogram/sample_predictions.png', dpi=300, bbox_inches='tight')
print("Örnek tahminler kaydedildi: results/melspectrogram/sample_predictions.png")
plt.close()

print("\n" + "="*60)
print("Mel-Spectrogram CNN Model Eğitimi Tamamlandı!")
print("="*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Tüm sonuçlar 'results/melspectrogram/' klasörüne kaydedildi.")
print("="*60)
