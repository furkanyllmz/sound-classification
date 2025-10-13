import os
import sys
import json
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, top_k_accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
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
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    BatchNormalization, Input, Resizing
)

# ============ GPU AYARLARI ============
# Apple Silicon (M1/M2/M3) ve NVIDIA GPU desteği
print("\n" + "="*60)
print("GPU Kontrol Ediliyor...")
print("="*60)

# Apple Metal GPU kontrolü
try:
    # Metal GPU'yu kontrol et
    metal_devices = tf.config.list_logical_devices('GPU')
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if physical_devices or metal_devices:
        print(f"✅ Apple Silicon GPU bulundu!")
        
        # Metal bellek büyümesini etkinleştir
        if physical_devices:
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_gpu_growth(gpu, True)
                print(f"   Physical GPU sayısı: {len(physical_devices)}")
            except:
                pass
                
        if metal_devices:
            print(f"   Metal GPU sayısı: {len(metal_devices)}")
            
        # Apple Silicon için mixed precision önerilmez
        print("✅ Metal backend ile GPU acceleration aktif")
        
        # TensorFlow'un GPU kullandığını doğrula
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.reduce_sum(test_tensor)
        print("✅ GPU testi başarılı")
        
    else:
        print("⚠️  GPU bulunamadı, CPU kullanılacak")
        
except Exception as e:
    print(f"❌ GPU kontrol hatası: {e}")
    print("⚠️  CPU kullanılacak")

print("="*60)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import csv

# =================== AYARLAR ===================
MODEL_NAME = 'DenseNet121'  # Komuttan da alabilir
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]

TARGET_SIZE = (224, 224)  # Base modeller için güvenli hedef boyut
RESULTS_DIR = f'results/mfcc_{MODEL_NAME.lower()}'
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"Seçilen Model: {MODEL_NAME}")
print(f"{'='*60}\n")

MODELS = {
    'VGG16': VGG16, 'VGG19': VGG19,
    'ResNet50': ResNet50, 'ResNet101': ResNet101, 'ResNet152': ResNet152,
    'InceptionV3': InceptionV3, 'InceptionResNetV2': InceptionResNetV2,
    'MobileNet': MobileNet, 'MobileNetV2': MobileNetV2,
    'DenseNet121': DenseNet121, 'DenseNet169': DenseNet169, 'DenseNet201': DenseNet201,
    'EfficientNetB0': EfficientNetB0, 'EfficientNetB1': EfficientNetB1,
    'EfficientNetB2': EfficientNetB2, 'EfficientNetB3': EfficientNetB3,
    'Xception': Xception
}

if MODEL_NAME not in MODELS:
    print(f"HATA: '{MODEL_NAME}' geçerli bir model değil!")
    print(f"Kullanılabilir modeller: {list(MODELS.keys())}")
    sys.exit(1)

# =================== MFCC ÇIKARIM ===================
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
        print(f"MFCC çıkarma hatası {file_path}: {e}")
        return None

def load_dataset(dataset_path='dataset'):
    """Tüm ses dosyalarını yükler ve MFCC özelliklerini çıkarır"""
    features, labels = [], []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()

    print(f"Bulunan sınıflar: {classes}")
    print(f"Toplam sınıf sayısı: {len(classes)}")

    for label in classes:
        class_path = os.path.join(dataset_path, label)
        print(f"\n{label} sınıfı işleniyor...")

        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        print(f"  {label} klasöründe {len(wav_files)} wav dosyası bulundu")

        count, success_count = 0, 0
        for filename in os.listdir(class_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_path, filename)
                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
                    success_count += 1
                count += 1
                if count % 100 == 0:
                    print(f"  {count} dosya işlendi, {success_count} başarılı...")

        print(f"{label}: Toplam {count} dosya işlendi, {success_count} başarılı")

    print(f"\nToplam yüklenen örnek sayısı: {len(features)}")
    print(f"Benzersiz sınıflar: {list(set(labels))}")
    print(f"Yüklenen sınıf sayısı: {len(set(labels))}")

    return np.array(features), np.array(labels), classes

print("Veri seti yükleniyor...")
X, y, class_names = load_dataset()

# =================== LABEL ENCODING ===================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# VGG/ResNet vb. RGB formatı beklediği için 3 kanal yap
X_rgb = np.repeat(X[..., np.newaxis], 3, -1)  # (40, 174, 3)

print(f"\nVeri seti şekli: {X_rgb.shape}")
print(f"Etiket şekli: {y_categorical.shape}")
print(f"Sınıflar: {class_names}")

# =================== TRAIN/VAL/TEST BÖLME (DÜZELTİLDİ) ===================
# stratify 1D olmalı -> y_encoded
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X_rgb, y_categorical, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Val bölmesi: stratify için y_train_enc (1D) veya y_train'den argmax
X_train, X_val, y_train, y_val, y_train_enc, y_val_enc = train_test_split(
    X_train, y_train, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# =================== MODEL OLUŞTURMA ===================
def create_transfer_learning_model(model_name, input_shape, num_classes, trainable_layers=4):
    """Transfer learning modeli oluşturur"""
    inputs = Input(shape=input_shape)
    # Bazı uygulama modelleri minimum 75x75 ister -> güvenli amaçla resize
    x = Resizing(TARGET_SIZE[0], TARGET_SIZE[1])(inputs)

    base_model_class = MODELS[model_name]
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )

    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    print(f"\nBase Model: {model_name}")
    print(f"Toplam katman sayısı: {len(base_model.layers)}")
    print(f"Eğitilebilir katman sayısı: {trainable_layers}")
    print(f"Dondurulmuş katman sayısı: {len(base_model.layers) - trainable_layers}")

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

model = create_transfer_learning_model(MODEL_NAME, X_train.shape[1:], len(class_names), trainable_layers=4)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        TopKCategoricalAccuracy(k=2, name='top2_acc')  # EK: Top-2 Accuracy
    ]
)

print("\nModel Yapısı:")
model.summary()

# =================== CALLBACKS ===================
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
csv_logger = CSVLogger(f'{RESULTS_DIR}/history.csv', separator=',', append=False)

# =================== EĞİTİM ===================
print(f"\n{MODEL_NAME} Transfer Learning model eğitimi başlıyor...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, reduce_lr, csv_logger],
    verbose=1
)

# =================== KAYIT ===================
model_path = f'{RESULTS_DIR}/{MODEL_NAME.lower()}_model.h5'
model.save(model_path)
print(f"\nModel kaydedildi: {model_path}")

# =================== TEST DEĞERLENDİRME ===================
print("\nTest verisi üzerinde değerlendirme...")
evaluation_results = model.evaluate(X_test, y_test, verbose=0)
if isinstance(evaluation_results, list):
    test_loss = evaluation_results[0]
    test_accuracy = evaluation_results[1] if len(evaluation_results) > 1 else 0.0
else:
    test_loss = evaluation_results
    test_accuracy = 0.0
    
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Tahminler
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# =================== EK METRİKLER ===================
print("\nEk metrikler hesaplanıyor (test seti)...")
test_top2_acc = top_k_accuracy_score(y_true_classes, y_pred, k=2)
bal_acc       = balanced_accuracy_score(y_true_classes, y_pred_classes)
kappa         = cohen_kappa_score(y_true_classes, y_pred_classes)
mcc           = matthews_corrcoef(y_true_classes, y_pred_classes)

print(f"Top-2 Accuracy: {test_top2_acc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Matthews Corrcoef (MCC): {mcc:.4f}")

# =================== SONUÇLARI KAYDET ===================
results = {
    'model_type': f'{MODEL_NAME} Transfer Learning',
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'top2_accuracy': float(test_top2_acc),
    'balanced_accuracy': float(bal_acc),
    'cohen_kappa': float(kappa),
    'mcc': float(mcc),
    'class_names': class_names,
    'classification_report': classification_report(
        y_true_classes, y_pred_classes,
        target_names=class_names, output_dict=True
    )
}
with open(f'{RESULTS_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# =================== GÖRSELLEŞTİRMELER ===================
# 1) Training & Validation Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'{MODEL_NAME} Transfer Learning: Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'{MODEL_NAME} Transfer Learning: Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
print(f"Training history grafiği kaydedildi: {RESULTS_DIR}/training_history.png")
plt.close()

# 2) Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_NAME} Transfer Learning: Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"Confusion matrix kaydedildi: {RESULTS_DIR}/confusion_matrix.png")
plt.close()

# 3) Normalized Confusion Matrix
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'{MODEL_NAME} Transfer Learning: Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label'); plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix kaydedildi: {RESULTS_DIR}/confusion_matrix_normalized.png")
plt.close()

# 4) Class-wise Accuracy Bar Plot
class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, class_accuracy, color='steelblue', edgecolor='black')
plt.title(f'{MODEL_NAME} Transfer Learning: Class-wise Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Class'); plt.ylabel('Accuracy'); plt.ylim([0, 1.1]); plt.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/class_accuracy.png', dpi=300, bbox_inches='tight')
print(f"Class accuracy grafiği kaydedildi: {RESULTS_DIR}/class_accuracy.png")
plt.close()

# 5) Learning Rate Schedule (varsa)
if 'lr' in history.history:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['lr'])
    plt.title(f'{MODEL_NAME} Transfer Learning: Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.yscale('log'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/learning_rate.png', dpi=300, bbox_inches='tight')
    print(f"Learning rate grafiği kaydedildi: {RESULTS_DIR}/learning_rate.png")
    plt.close()

# 6) Örnek MFCC görselleştirmesi
print("\nÖrnek MFCC görselleştirmeleri oluşturuluyor...")
n_show = min(6, len(X_test))
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i in range(n_show):
    axes[i].imshow(X_test[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f'True: {class_names[y_true_classes[i]]}\nPred: {class_names[y_pred_classes[i]]}')
    axes[i].set_xlabel('Time'); axes[i].set_ylabel('MFCC Coefficients')
plt.suptitle(f'{MODEL_NAME} Transfer Learning: Sample Predictions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/sample_predictions.png', dpi=300, bbox_inches='tight')
print(f"Örnek tahminler kaydedildi: {RESULTS_DIR}/sample_predictions.png")
plt.close()

# =================== EPOCH-BAZLI TABLO/CSV/ÇİZİM ===================
epochs = list(range(1, len(history.history['accuracy']) + 1))
epoch_rows = []
for i, ep in enumerate(epochs):
    row = {
        'epoch': ep,
        'accuracy': float(history.history['accuracy'][i]),
        'val_accuracy': float(history.history['val_accuracy'][i]),
        'loss': float(history.history['loss'][i]),
        'val_loss': float(history.history['val_loss'][i]),
    }
    if 'top2_acc' in history.history:
        row['top2_acc'] = float(history.history['top2_acc'][i])
    if 'val_top2_acc' in history.history:
        row['val_top2_acc'] = float(history.history['val_top2_acc'][i])
    epoch_rows.append(row)

print("\nEpoch |  acc     val_acc   loss     val_loss   (top2/val_top2 varsa)")
for r in epoch_rows:
    extra = ""
    if 'top2_acc' in r:
        extra += f"  top2={r['top2_acc']:.4f}"
    if 'val_top2_acc' in r:
        extra += f"  val_top2={r['val_top2_acc']:.4f}"
    print(f"{r['epoch']:>5} | {r['accuracy']:.4f}  {r['val_accuracy']:.4f}  {r['loss']:.4f}  {r['val_loss']:.4f}{extra}")

with open(f'{RESULTS_DIR}/epoch_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
    writer.writeheader()
    writer.writerows(epoch_rows)
print(f"\nEpoch metrikleri kaydedildi: {RESULTS_DIR}/epoch_metrics.csv")

plt.figure(figsize=(10, 6))
plt.plot(epochs, [r['accuracy'] for r in epoch_rows], label='Train Accuracy')
plt.plot(epochs, [r['val_accuracy'] for r in epoch_rows], label='Validation Accuracy')
plt.title(f'{MODEL_NAME} - Accuracy by Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/accuracy_by_epoch.png', dpi=300, bbox_inches='tight')
print(f"Accuracy by epoch grafiği kaydedildi: {RESULTS_DIR}/accuracy_by_epoch.png")
plt.close()

print("\n" + "="*60)
print(f"{MODEL_NAME} Transfer Learning Model Eğitimi Tamamlandı!")
print("="*60)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Tüm sonuçlar '{RESULTS_DIR}/' klasörüne kaydedildi.")
print("="*60)
