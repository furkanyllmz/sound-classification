"""
YAMNet Transfer Learning - IMPROVED VERSION
Daha iyi sonuçlar için optimize edilmiş mimari ve hiperparametreler
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import librosa

# ============ CONFIGURATION ============
# Farklı mimarileri denemek için aşağıdaki değerleri değiştirin

CONFIG = {
    'architecture': 'residual',  # 'shallow', 'medium', 'deep', 'residual', 'ensemble'
    'dropout_rate': 0.4,     # 0.3-0.5 arası
    'learning_rate': 5e-4,   # 1e-4, 5e-4, 1e-3
    'batch_size': 32,        # 16, 32, 64
    'epochs': 100,
    'l2_reg': 1e-4,         # L2 regularization
    'use_batch_norm': True,
    'use_class_weights': True  # Dengesiz sınıflar için
}

print("\n" + "="*70)
print(f"YAMNet Transfer Learning - IMPROVED")
print("="*70)
print(f"\n📋 Configuration:")
for key, value in CONFIG.items():
    print(f"  {key:20s}: {value}")
print("\n" + "="*70 + "\n")

# Sonuçları kaydetmek için klasör oluştur
RESULTS_DIR = f'results/yamnet_improved_{CONFIG["architecture"]}'
os.makedirs(RESULTS_DIR, exist_ok=True)

# YAMNet model URL
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

print("YAMNet modeli yükleniyor...")
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
print("✓ YAMNet modeli yüklendi!")

# YAMNet için ses yükleme ve preprocessing
def load_audio_for_yamnet(file_path):
    """YAMNet için ses dosyasını yükler ve hazırlar"""
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        max_length = 4 * 16000
        if len(audio) < max_length:
            audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
        else:
            audio = audio[:max_length]
        return audio
    except Exception as e:
        print(f"Hata {file_path}: {e}")
        return None

def extract_yamnet_embeddings(audio):
    """YAMNet'ten embedding vektörlerini çıkarır"""
    audio_tensor = tf.cast(audio, tf.float32)
    scores, embeddings, spectrogram = yamnet_model(audio_tensor)
    embedding_mean = tf.reduce_mean(embeddings, axis=0)
    return embedding_mean.numpy()

def load_dataset_yamnet(dataset_path='dataset'):
    """Tüm ses dosyalarını yükler ve YAMNet embeddings çıkarır"""
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
                audio = load_audio_for_yamnet(file_path)
                if audio is not None:
                    embedding = extract_yamnet_embeddings(audio)
                    features.append(embedding)
                    labels.append(label)
                    count += 1
                    if count % 100 == 0:
                        print(f"  {count} dosya işlendi...")

        print(f"✓ {label}: Toplam {count} dosya yüklendi\n")

    return np.array(features), np.array(labels), classes

print("Veri seti yükleniyor ve YAMNet embeddings çıkarılıyor...")
X, y, class_names = load_dataset_yamnet()

# Label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"\nVeri seti şekli: {X.shape}")
print(f"Etiket şekli: {y_categorical.shape}")
print(f"Sınıflar: {class_names}")

# Normalizasyon
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / (std + 1e-8)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTrain set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Class weights (dengesiz sınıflar için)
if CONFIG['use_class_weights']:
    class_counts = np.sum(y_train, axis=0)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_names) * count)
                    for i, count in enumerate(class_counts)}
    print(f"\n⚖️  Class weights hesaplandı:")
    for i, (cls, weight) in enumerate(zip(class_names, class_weights.values())):
        print(f"  {cls:20s}: {weight:.3f}")
else:
    class_weights = None

# ============ MODEL MİMARİLERİ ============

def create_shallow_model(input_shape, num_classes, config):
    """Basit ve hızlı mimari"""
    inputs = Input(shape=input_shape)

    x = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='YAMNet_Shallow')

def create_medium_model(input_shape, num_classes, config):
    """Orta derinlikte mimari (orijinal)"""
    inputs = Input(shape=input_shape)

    x = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.7)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='YAMNet_Medium')

def create_deep_model(input_shape, num_classes, config):
    """Derin mimari - daha fazla kapasite"""
    inputs = Input(shape=input_shape)

    x = Dense(768, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(384, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.8)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.7)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='YAMNet_Deep')

def create_residual_model(input_shape, num_classes, config):
    """Residual connection'lı mimari"""
    inputs = Input(shape=input_shape)

    # İlk blok
    x = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    # Residual blok 1
    residual = x
    x = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    x = tf.keras.layers.Add()([x, residual])  # Skip connection
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    # Downsampling
    x = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.7)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='YAMNet_Residual')

def create_ensemble_model(input_shape, num_classes, config):
    """Birden fazla branch ile ensemble benzeri mimari"""
    inputs = Input(shape=input_shape)

    # Branch 1: Wide
    branch1 = Dense(512, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        branch1 = BatchNormalization()(branch1)
    branch1 = Dropout(config['dropout_rate'])(branch1)
    branch1 = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(branch1)

    # Branch 2: Deep
    branch2 = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(inputs)
    if config['use_batch_norm']:
        branch2 = BatchNormalization()(branch2)
    branch2 = Dropout(config['dropout_rate'])(branch2)
    branch2 = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(branch2)
    if config['use_batch_norm']:
        branch2 = BatchNormalization()(branch2)
    branch2 = Dropout(config['dropout_rate'] * 0.7)(branch2)
    branch2 = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(branch2)

    # Concatenate
    x = Concatenate()([branch1, branch2])
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(config['l2_reg']))(x)
    if config['use_batch_norm']:
        x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'] * 0.7)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='YAMNet_Ensemble')

# Model seçimi
MODEL_BUILDERS = {
    'shallow': create_shallow_model,
    'medium': create_medium_model,
    'deep': create_deep_model,
    'residual': create_residual_model,
    'ensemble': create_ensemble_model
}

print(f"\n🏗️  Model oluşturuluyor: {CONFIG['architecture']}")
model_builder = MODEL_BUILDERS[CONFIG['architecture']]
model = model_builder(X_train.shape[1:], len(class_names), CONFIG)

model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Model Yapısı:")
model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
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

# Model eğitimi
print(f"\n🚀 Model eğitimi başlıyor...")
print("="*70)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# En iyi modeli yükle
model.load_weights(f'{RESULTS_DIR}/best_model.h5')
model.save(f'{RESULTS_DIR}/yamnet_improved_model.h5')
print(f"\n✓ Model kaydedildi: {RESULTS_DIR}/yamnet_improved_model.h5")

# Test değerlendirme
print("\n📈 Test verisi üzerinde değerlendirme...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")

# Tahminler
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification Report
print("\n" + "="*70)
print("Classification Report:")
print("="*70)
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# ============ EK METRİKLER HESAPLA ============

# Per-class metrics
per_class_metrics = {}
for i, cls in enumerate(class_names):
    cls_indices = y_true_classes == i
    if cls_indices.sum() > 0:
        cls_pred = y_pred_classes[cls_indices]
        cls_true = y_true_classes[cls_indices]

        # True Positives, False Positives, True Negatives, False Negatives
        tp = ((cls_pred == i) & (cls_true == i)).sum()
        fp = ((cls_pred == i) & (cls_true != i)).sum()
        tn = ((cls_pred != i) & (cls_true != i)).sum()
        fn = ((cls_pred != i) & (cls_true == i)).sum()

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Balanced Accuracy
        report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
        sensitivity = report[cls]['recall']
        balanced_acc = (sensitivity + specificity) / 2

        per_class_metrics[cls] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'balanced_accuracy': float(balanced_acc),
            'error_rate': 1 - sensitivity
        }

# Overall additional metrics
macro_balanced_acc = np.mean([per_class_metrics[cls]['balanced_accuracy'] for cls in class_names])
macro_specificity = np.mean([per_class_metrics[cls]['specificity'] for cls in class_names])
macro_error_rate = np.mean([per_class_metrics[cls]['error_rate'] for cls in class_names])

# Top-k accuracy (top-2, top-3)
top_2_correct = sum(y_true_classes[i] in np.argsort(y_pred[i])[-2:] for i in range(len(y_test)))
top_3_correct = sum(y_true_classes[i] in np.argsort(y_pred[i])[-3:] for i in range(len(y_test)))
top_2_accuracy = top_2_correct / len(y_test)
top_3_accuracy = top_3_correct / len(y_test)

# Confidence metrics
confidence_scores = np.max(y_pred, axis=1)
avg_confidence = np.mean(confidence_scores)
correct_confidence = np.mean(confidence_scores[y_pred_classes == y_true_classes])
incorrect_confidence = np.mean(confidence_scores[y_pred_classes != y_true_classes])

# Sonuçları kaydet
results = {
    'model_type': f'YAMNet Improved - {CONFIG["architecture"]}',
    'base_model': 'YAMNet (AudioSet pretrained)',
    'configuration': CONFIG,
    'test_loss': float(test_loss),
    'test_accuracy': float(test_accuracy),
    'class_names': class_names,
    'num_classes': len(class_names),
    'classification_report': classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True),

    # Ek metrikler
    'per_class_detailed_metrics': per_class_metrics,
    'additional_metrics': {
        'top_2_accuracy': float(top_2_accuracy),
        'top_3_accuracy': float(top_3_accuracy),
        'macro_balanced_accuracy': float(macro_balanced_acc),
        'macro_specificity': float(macro_specificity),
        'macro_error_rate': float(macro_error_rate),
        'average_confidence': float(avg_confidence),
        'correct_predictions_confidence': float(correct_confidence),
        'incorrect_predictions_confidence': float(incorrect_confidence),
        'confidence_gap': float(correct_confidence - incorrect_confidence)
    },
    'training_info': {
        'total_epochs': len(history.history['loss']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'best_val_loss': float(min(history.history['val_loss'])),
        'final_learning_rate': float(history.history.get('lr', [CONFIG['learning_rate']])[-1]) if 'lr' in history.history else CONFIG['learning_rate']
    }
}

with open(f'{RESULTS_DIR}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Görselleştirmeler (basitleştirilmiş)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title(f'Accuracy - {CONFIG["architecture"]}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title(f'Loss - {CONFIG["architecture"]}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300)
print(f"\n✓ Training history kaydedildi")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {CONFIG["architecture"]}')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=300)
print(f"✓ Confusion matrix kaydedildi")
plt.close()

# ============ EK GÖRSELLEŞTİRMELER ============

# 1. Per-class detailed metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1.1 Sensitivity vs Specificity
sensitivities = [per_class_metrics[cls]['sensitivity'] for cls in class_names]
specificities = [per_class_metrics[cls]['specificity'] for cls in class_names]

axes[0, 0].scatter(sensitivities, specificities, s=200, c=range(len(class_names)),
                  cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
for i, cls in enumerate(class_names):
    axes[0, 0].annotate(cls, (sensitivities[i], specificities[i]),
                       fontsize=8, ha='center', va='center')
axes[0, 0].set_xlabel('Sensitivity (Recall)', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Specificity', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Sensitivity vs Specificity', fontweight='bold', fontsize=12)
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim([0.85, 1.0])
axes[0, 0].set_ylim([0.85, 1.0])

# 1.2 Balanced Accuracy
balanced_accs = [per_class_metrics[cls]['balanced_accuracy'] for cls in class_names]
colors_ba = plt.cm.RdYlGn(balanced_accs)
bars = axes[0, 1].barh(class_names, balanced_accs, color=colors_ba, edgecolor='black', linewidth=1)
axes[0, 1].set_xlabel('Balanced Accuracy', fontweight='bold', fontsize=11)
axes[0, 1].set_title('Balanced Accuracy per Class', fontweight='bold', fontsize=12)
axes[0, 1].set_xlim([0.85, 1.0])
axes[0, 1].grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, balanced_accs):
    axes[0, 1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.2%}', va='center', fontsize=9, fontweight='bold')

# 1.3 Error Rate
error_rates = [per_class_metrics[cls]['error_rate'] for cls in class_names]
bars = axes[1, 0].barh(class_names, error_rates, color='coral', alpha=0.7,
                      edgecolor='black', linewidth=1)
axes[1, 0].set_xlabel('Error Rate', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Error Rate per Class', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, error_rates):
    if val > 0.01:
        axes[1, 0].text(val + 0.003, bar.get_y() + bar.get_height()/2,
                       f'{val:.2%}', va='center', fontsize=9, fontweight='bold')

# 1.4 TP/FP/TN/FN breakdown for most challenging class
worst_cls_idx = np.argmin(balanced_accs)
worst_cls = class_names[worst_cls_idx]
tp = per_class_metrics[worst_cls]['true_positives']
fp = per_class_metrics[worst_cls]['false_positives']
tn = per_class_metrics[worst_cls]['true_negatives']
fn = per_class_metrics[worst_cls]['false_negatives']

confusion_data = [tp, fp, fn, tn]
confusion_labels = ['True\nPositive', 'False\nPositive', 'False\nNegative', 'True\nNegative']
confusion_colors = ['#4CAF50', '#FF9800', '#FF5722', '#2196F3']

bars = axes[1, 1].bar(range(4), confusion_data, color=confusion_colors, alpha=0.7,
                     edgecolor='black', linewidth=2)
axes[1, 1].set_xticks(range(4))
axes[1, 1].set_xticklabels(confusion_labels, fontsize=9)
axes[1, 1].set_ylabel('Count', fontweight='bold', fontsize=11)
axes[1, 1].set_title(f'Confusion Breakdown: {worst_cls}\n(Most Challenging Class)',
                    fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, confusion_data):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle(f'Detailed Performance Metrics - {CONFIG["architecture"].upper()}',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/detailed_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Detailed metrics kaydedildi")
plt.close()

# 2. Confidence Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 2.1 Confidence distribution
axes[0].hist(confidence_scores, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
axes[0].axvline(avg_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_confidence:.2%}')
axes[0].set_xlabel('Confidence Score', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Confidence Score Distribution', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2.2 Correct vs Incorrect confidence
conf_data = [correct_confidence, incorrect_confidence]
conf_labels = ['Correct\nPredictions', 'Incorrect\nPredictions']
conf_colors = ['green', 'red']
bars = axes[1].bar(range(2), conf_data, color=conf_colors, alpha=0.7,
                  edgecolor='black', linewidth=2, width=0.6)
axes[1].set_xticks(range(2))
axes[1].set_xticklabels(conf_labels)
axes[1].set_ylabel('Average Confidence', fontweight='bold')
axes[1].set_title('Confidence: Correct vs Incorrect', fontweight='bold', fontsize=12)
axes[1].set_ylim([0, 1.0])
axes[1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, conf_data):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 2.3 Top-k Accuracy
topk_data = [test_accuracy, top_2_accuracy, top_3_accuracy]
topk_labels = ['Top-1', 'Top-2', 'Top-3']
topk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = axes[2].bar(range(3), topk_data, color=topk_colors, alpha=0.7,
                  edgecolor='black', linewidth=2, width=0.6)
axes[2].set_xticks(range(3))
axes[2].set_xticklabels(topk_labels)
axes[2].set_ylabel('Accuracy', fontweight='bold')
axes[2].set_title('Top-k Accuracy', fontweight='bold', fontsize=12)
axes[2].set_ylim([0.8, 1.0])
axes[2].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, topk_data):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Confidence & Top-k Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confidence_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Confidence analysis kaydedildi")
plt.close()

# 3. Training info visualization
if 'lr' in history.history:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Learning rate schedule
    axes[0].plot(history.history['lr'], linewidth=2, color='orangered')
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Learning Rate', fontweight='bold')
    axes[0].set_title('Learning Rate Schedule', fontweight='bold', fontsize=12)
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    # Val accuracy vs LR
    axes[1].plot(history.history['lr'], history.history['val_accuracy'],
                marker='o', markersize=4, alpha=0.6, linewidth=1)
    axes[1].set_xlabel('Learning Rate', fontweight='bold')
    axes[1].set_ylabel('Validation Accuracy', fontweight='bold')
    axes[1].set_title('Validation Accuracy vs Learning Rate', fontweight='bold', fontsize=12)
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/learning_rate_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Learning rate analysis kaydedildi")
    plt.close()

print("\n" + "="*70)
print(f"✅ YAMNet IMPROVED Eğitimi Tamamlandı!")
print("="*70)
print(f"\n📊 PERFORMANS METRİKLERİ:")
print(f"  • Test Accuracy:           {test_accuracy:.2%}")
print(f"  • Test Loss:               {test_loss:.4f}")
print(f"  • Macro Balanced Accuracy: {macro_balanced_acc:.2%}")
print(f"  • Top-2 Accuracy:          {top_2_accuracy:.2%}")
print(f"  • Top-3 Accuracy:          {top_3_accuracy:.2%}")
print(f"  • Avg Confidence:          {avg_confidence:.2%}")
print(f"  • Confidence Gap:          {correct_confidence - incorrect_confidence:.2%}")
print(f"\n📂 Sonuçlar: {RESULTS_DIR}/")
print("="*70 + "\n")
