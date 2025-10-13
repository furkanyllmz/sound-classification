import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import librosa

# Sonuçları kaydetmek için klasör oluştur
RESULTS_DIR = 'results/yamnet_transfer'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "="*60)
print("YAMNet Transfer Learning - UrbanSound8K")
print("="*60 + "\n")

# YAMNet model URL
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

print("YAMNet modeli yükleniyor...")
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
print("✓ YAMNet modeli yüklendi!")

# YAMNet için ses yükleme ve preprocessing
def load_audio_for_yamnet(file_path):
    """
    YAMNet için ses dosyasını yükler ve hazırlar
    YAMNet 16kHz mono ses bekler
    """
    try:
        # Librosa ile yükle (YAMNet 16kHz bekler)
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # YAMNet maksimum 10 saniye işler, biz 4 saniye kullanacağız
        max_length = 4 * 16000  # 4 saniye
        if len(audio) < max_length:
            # Padding
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        else:
            # Truncate
            audio = audio[:max_length]

        return audio
    except Exception as e:
        print(f"Hata {file_path}: {e}")
        return None

def extract_yamnet_embeddings(audio):
    """
    YAMNet'ten embedding vektörlerini çıkarır
    """
    # YAMNet'e float32 tensor olarak ver
    audio_tensor = tf.cast(audio, tf.float32)

    # YAMNet inference
    scores, embeddings, spectrogram = yamnet_model(audio_tensor)

    # Embeddings shape: (N, 1024) - N frame sayısı
    # Tüm framelerin ortalamasını al
    embedding_mean = tf.reduce_mean(embeddings, axis=0)

    return embedding_mean.numpy()

# Veri setini yükle ve YAMNet embeddings çıkar
def load_dataset_yamnet(dataset_path='dataset'):
    """
    Tüm ses dosyalarını yükler ve YAMNet embeddings çıkarır
    """
    features = []
    labels = []

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()

    print(f"Bulunan sınıflar: {classes}\n")

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        print(f"{label} sınıfı işleniyor...")
        count = 0

        for filename in os.listdir(class_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_path, filename)

                # Ses dosyasını yükle
                audio = load_audio_for_yamnet(file_path)
                if audio is not None:
                    # YAMNet embeddings çıkar
                    embedding = extract_yamnet_embeddings(audio)
                    features.append(embedding)
                    labels.append(label)
                    count += 1

                    if count % 100 == 0:
                        print(f"  {count} dosya işlendi...")

        print(f"✓ {label}: Toplam {count} dosya yüklendi\n")

    return np.array(features), np.array(labels), classes

print("Veri seti yükleniyor ve YAMNet embeddings çıkarılıyor...")
print("Bu işlem birkaç dakika sürebilir...\n")
X, y, class_names = load_dataset_yamnet()

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"\nVeri seti şekli: {X.shape}")
print(f"Etiket şekli: {y_categorical.shape}")
print(f"Sınıflar: {class_names}")
print(f"Sınıf sayısı: {len(class_names)}")

# Normalizasyon
print("\nVeri normalizasyonu yapılıyor...")
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / (std + 1e-8)

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

# YAMNet Transfer Learning Modeli Oluştur
def create_yamnet_transfer_model(input_shape, num_classes):
    """
    YAMNet embeddings üzerine classification head ekler
    """
    inputs = Input(shape=input_shape, name='input_embeddings')

    # Classification head
    x = Dense(512, activation='relu', name='dense_1')(inputs)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = Dense(128, activation='relu', name='dense_3')(x)
    x = Dropout(0.3, name='dropout_3')(x)

    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='YAMNet_Transfer_Learning')

    return model

# Modeli oluştur
print("\nYAMNet Transfer Learning modeli oluşturuluyor...")
model = create_yamnet_transfer_model(X_train.shape[1:], len(class_names))

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Yapısı:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    f'{RESULTS_DIR}/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Modeli eğit
print("\nYAMNet Transfer Learning model eğitimi başlıyor...")
print("="*60)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# En iyi modeli yükle
print("\nEn iyi model yükleniyor...")
model.load_weights(f'{RESULTS_DIR}/best_model.h5')

# Modeli kaydet
model.save(f'{RESULTS_DIR}/yamnet_transfer_model.h5')
print(f"✓ Model kaydedildi: {RESULTS_DIR}/yamnet_transfer_model.h5")

# Test verisi üzerinde değerlendirme
print("\nTest verisi üzerinde değerlendirme...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Tahminler
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Sonuçları kaydet
results = {
    'model_type': 'YAMNet Transfer Learning',
    'base_model': 'YAMNet (AudioSet pretrained)',
    'frozen_layers': 'All YAMNet layers frozen',
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'class_names': class_names,
    'num_classes': len(class_names),
    'embedding_dim': 1024,
    'total_samples': len(X),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'classification_report': classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
}

with open(f'{RESULTS_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✓ Sonuçlar kaydedildi: {RESULTS_DIR}/results.json")

# ============ GÖRSELLEŞTIRMELER ============

print("\nGörselleştirmeler oluşturuluyor...")

# 1. Training & Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('YAMNet Transfer Learning: Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 2. Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('YAMNet Transfer Learning: Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Training history: {RESULTS_DIR}/training_history.png")
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, annot_kws={'size': 10})
plt.title('YAMNet Transfer Learning: Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix: {RESULTS_DIR}/confusion_matrix.png")
plt.close()

# 4. Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}, annot_kws={'size': 10})
plt.title('YAMNet Transfer Learning: Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(f"✓ Normalized confusion matrix: {RESULTS_DIR}/confusion_matrix_normalized.png")
plt.close()

# 5. Class-wise Accuracy Bar Plot
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
bars = plt.bar(class_names, class_accuracy, color=colors, edgecolor='black', linewidth=1.5)
plt.title('YAMNet Transfer Learning: Class-wise Accuracy', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

# Bar'ların üzerine değerleri yaz
for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"✓ Class accuracy: {RESULTS_DIR}/class_accuracy.png")
plt.close()

# 6. Learning Rate Schedule
if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'], linewidth=2, color='orangered')
    plt.title('YAMNet Transfer Learning: Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/learning_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Learning rate: {RESULTS_DIR}/learning_rate.png")
    plt.close()

# 7. Per-class Precision, Recall, F1-Score
report_dict = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
metric_values = {metric: [report_dict[cls][metric] for cls in class_names] for metric in metrics}

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, metric_values[metric], width, label=metric.capitalize(), alpha=0.8)

ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('YAMNet Transfer Learning: Per-Class Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/per_class_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Per-class metrics: {RESULTS_DIR}/per_class_metrics.png")
plt.close()

# 8. Model Summary Text File
with open(f'{RESULTS_DIR}/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print(f"✓ Model summary: {RESULTS_DIR}/model_summary.txt")

print("\n" + "="*60)
print("YAMNet Transfer Learning Model Eğitimi Tamamlandı!")
print("="*60)
print(f"✓ Test Accuracy: {test_accuracy:.2%}")
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Tüm sonuçlar '{RESULTS_DIR}/' klasörüne kaydedildi.")
print("="*60 + "\n")

# En iyi ve en kötü sınıfları göster
best_class_idx = np.argmax(class_accuracy)
worst_class_idx = np.argmin(class_accuracy)

print("📊 Sınıf Performans Özeti:")
print(f"  ✅ En iyi: {class_names[best_class_idx]} ({class_accuracy[best_class_idx]:.1%})")
print(f"  ⚠️  En kötü: {class_names[worst_class_idx]} ({class_accuracy[worst_class_idx]:.1%})")
print(f"  📈 Ortalama: {class_accuracy.mean():.1%}")
print("\n")
