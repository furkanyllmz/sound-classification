import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import (
    VGG16, VGG19,
    ResNet50, ResNet101, ResNet152,
    InceptionV3, InceptionResNetV2,
    MobileNet, MobileNetV2,
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    Xception
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import sys

# ============ MODEL SEÇİMİ ============
# Kullanılabilir modeller:
# 'VGG16', 'VGG19'
# 'ResNet50', 'ResNet101', 'ResNet152'
# 'InceptionV3', 'InceptionResNetV2'
# 'MobileNet', 'MobileNetV2'
# 'DenseNet121', 'DenseNet169', 'DenseNet201'
# 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3'
# 'Xception'

MODEL_NAME = 'VGG16'  # Buradan model ismini değiştirebilirsiniz

# Komut satırından model adı alınabilir
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]

print(f"\n{'='*60}")
print(f"Seçilen Model: {MODEL_NAME}")
print(f"{'='*60}\n")

# Model sözlüğü
MODELS = {
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'InceptionV3': InceptionV3,
    'InceptionResNetV2': InceptionResNetV2,
    'MobileNet': MobileNet,
    'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201,
    'EfficientNetB0': EfficientNetB0,
    'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2,
    'EfficientNetB3': EfficientNetB3,
    'Xception': Xception
}

# Model kontrolü
if MODEL_NAME not in MODELS:
    print(f"HATA: '{MODEL_NAME}' geçerli bir model değil!")
    print(f"Kullanılabilir modeller: {list(MODELS.keys())}")
    sys.exit(1)

# Sonuçları kaydetmek için klasör oluştur
RESULTS_DIR = f'results/mfcc_{MODEL_NAME.lower()}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Veri setini yükle ve MFCC özellikleri çıkar
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

# Veri setini hazırla
def load_dataset(dataset_path='dataset'):
    """Tüm ses dosyalarını yükler ve MFCC özelliklerini çıkarır"""
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
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
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

# VGG16 RGB formatı beklediği için MFCC'yi 3 kanala çevir
X_rgb = np.repeat(X[..., np.newaxis], 3, -1)

print(f"\nVeri seti şekli: {X_rgb.shape}")
print(f"Etiket şekli: {y_categorical.shape}")
print(f"Sınıflar: {class_names}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_rgb, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Transfer Learning Modeli Oluştur
def create_transfer_learning_model(model_name, input_shape, num_classes, trainable_layers=4):
    """Transfer learning modeli oluşturur"""

    # Input layer
    inputs = Input(shape=input_shape)

    # Base model (ImageNet ağırlıkları ile)
    base_model_class = MODELS[model_name]
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs
    )

    # Base model katmanlarını dondur (fine-tuning için son birkaç katman açılabilir)
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    print(f"\nBase Model: {model_name}")
    print(f"Toplam katman sayısı: {len(base_model.layers)}")
    print(f"Eğitilebilir katman sayısı: {trainable_layers}")
    print(f"Dondurulmuş katman sayısı: {len(base_model.layers) - trainable_layers}")

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model

# Modeli oluştur
model = create_transfer_learning_model(MODEL_NAME, X_train.shape[1:], len(class_names), trainable_layers=4)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
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

# Modeli eğit
print(f"\n{MODEL_NAME} Transfer Learning model eğitimi başlıyor...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Modeli kaydet
model_path = f'{RESULTS_DIR}/{MODEL_NAME.lower()}_model.h5'
model.save(model_path)
print(f"\nModel kaydedildi: {model_path}")

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
    'model_type': f'{MODEL_NAME} Transfer Learning',
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'class_names': class_names,
    'classification_report': classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
}

with open(f'{RESULTS_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# ============ GÖRSELLEŞTIRMELER ============

# 1. Training & Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'{MODEL_NAME} Transfer Learning: Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'{MODEL_NAME} Transfer Learning: Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
print(f"Training history grafiği kaydedildi: {RESULTS_DIR}/training_history.png")
plt.close()

# 3. Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_NAME} Transfer Learning: Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix kaydedildi: {RESULTS_DIR}/confusion_matrix.png")
plt.close()

# 4. Normalized Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_NAME} Transfer Learning: Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix kaydedildi: {RESULTS_DIR}/confusion_matrix_normalized.png")
plt.close()

# 5. Class-wise Accuracy Bar Plot
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracy, color='steelblue', edgecolor='black')
plt.title(f'{MODEL_NAME} Transfer Learning: Class-wise Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# Bar'ların üzerine değerleri yaz
for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Class accuracy grafiği kaydedildi: {RESULTS_DIR}/class_accuracy.png")
plt.close()

# 6. Learning Rate Schedule
if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'])
    plt.title(f'{MODEL_NAME} Transfer Learning: Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/learning_rate.png', dpi=300, bbox_inches='tight')
    print(f"Learning rate grafiği kaydedildi: {RESULTS_DIR}/learning_rate.png")
    plt.close()

# 7. Örnek MFCC görselleştirmesi (tek kanal olarak)
print("\nÖrnek MFCC görselleştirmeleri oluşturuluyor...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(min(6, len(X_test))):
    axes[i].imshow(X_test[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f'True: {class_names[y_true_classes[i]]}\nPred: {class_names[y_pred_classes[i]]}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('MFCC Coefficients')

plt.suptitle(f'{MODEL_NAME} Transfer Learning: Sample Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/sample_predictions.png', dpi=300, bbox_inches='tight')
print(f"Örnek tahminler kaydedildi: {RESULTS_DIR}/sample_predictions.png")
plt.close()

print("\n" + "="*60)
print(f"{MODEL_NAME} Transfer Learning Model Eğitimi Tamamlandı!")
print("="*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Tüm sonuçlar '{RESULTS_DIR}/' klasörüne kaydedildi.")
print("="*60)
