# Derin Öğrenme Tabanlı Ses Sınıflandırma ve Gelecek Uygulamaları

**Proje Ekibi:** Furkan Yılmaz, Aleyna Taşdemir, Edanur Arslan, Talip Kurt
**Danışman:** Doç. Dr. FATİH EKİNCİ
**Tarih:** Ekim 2025

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Veri Seti](#veri-seti)
- [Özellik Çıkarım Yöntemleri](#özellik-çıkarım-yöntemleri)
- [Model Mimarileri](#model-mimarileri)
- [Performans Sonuçları](#performans-sonuçları)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Gelecek Çalışmalar](#gelecek-çalışmalar)
- [Referanslar](#referanslar)

---

## Proje Özeti

Bu proje, ses sinyallerinin derin öğrenme tabanlı sınıflandırılmasına yönelik kapsamlı bir araştırma çalışmasıdır. **UrbanSound8K** veri seti kullanılarak şehir ortamındaki farklı ses türlerinin otomatik olarak tanınması sağlanmıştır.

### Sınıflandırılan Ses Türleri (10 Sınıf)
- Klima sesi (air_conditioner) - 1,000 örnek
- Araba kornası (car_horn) - 429 örnek
- Çocuk sesleri (children_playing) - 1,000 örnek
- Köpek havlaması (dog_bark) - 1,000 örnek
- Matkap sesi (drilling) - 1,000 örnek
- Motor sesi (engine_idling) - 1,000 örnek
- Silah sesi (gun_shot) - 374 örnek
- Kırıcı sesi (jackhammer) - 1,000 örnek
- Siren sesi (siren) - 929 örnek
- Sokak müziği (street_music) - 1,000 örnek

**Toplam Veri:** 8,732 ses dosyası

### Ana Bulgular

| Model | Özellik Temsili | Doğruluk | Öne Çıkan Avantaj |
|-------|-----------------|----------|-------------------|
| **PANNs CNN14 + MLP** | PANNs Embeddings | **94.7%** | En yüksek genel performans |
| **Mel-Spektrogram CNN** | Mel-Spektrogram | **94.2%** | Transient seslerde üstün |
| **MFCC CNN** | MFCC (40 katsayı) | **93.1%** | %15 daha hızlı, gürültüye dayanıklı |
| **ResNet50 + MFCC** | MFCC | **88.0%** | Transfer öğrenme temeli |

---

## Veri Seti

### UrbanSound8K Özellikleri

- **Kaynak:** UrbanSound8K Dataset (şehir ortamı ses örnekleri)
- **Toplam Örnek:** 8,732 ses dosyası
- **Ses Uzunluğu:** 4 saniye (sabit)
- **Örnekleme Frekansı:**
  - 44.1 kHz (standart)
  - 16 kHz (YAMNet için)
  - 32 kHz (bazı transfer öğrenme modelleri için)
- **Kanal:** Mono
- **Format:** WAV dosyaları
- **Normalizasyon:** Z-score normalizasyonu uygulandı

### Veri Bölünmesi
- **Eğitim:** %80
- **Doğrulama:** %10
- **Test:** %10
- **Strateji:** Sınıf dengesi korunarak katmanlı bölme

### Sınıf Dağılımı

```
air_conditioner:   1,000 (%11.4) - Sürekli, sabit ses
car_horn:            429 (%4.9)  - Kısa, keskin geçici ses
children_playing:  1,000 (%11.4) - Karmaşık, değişken
dog_bark:          1,000 (%11.4) - Periyodik geçici ses
drilling:          1,000 (%11.4) - Tekrarlayan, mekanik
engine_idling:     1,000 (%11.4) - Sürekli, sabit ses
gun_shot:            374 (%4.3)  - Ani, yüksek şiddetli
jackhammer:        1,000 (%11.4) - Tekrarlayan, mekanik
siren:               929 (%10.6) - Modüle edilmiş sürekli ses
street_music:      1,000 (%11.4) - Karmaşık, değişken
```

**Not:** car_horn ve gun_shot sınıfları daha az örneklidir, bu nedenle katmanlı bölme stratejisi kullanılmıştır.

---

## Özellik Çıkarım Yöntemleri

### 1. Mel-Spektrogram

Mel-Spektrogram, insan kulağının algısal özelliklerini modelleyen ve ses sinyallerinin zaman-frekans temsilini sağlayan bir tekniktir.

#### Teknik Detaylar
- **Mel Band Sayısı:** 128
- **FFT Boyutu:** 1024
- **Hop Length:** 256
- **Frekans Aralığı:** 20 Hz - 8000 Hz
- **İşlem Adımları:**
  1. STFT (Short-Time Fourier Transform) uygulanması
  2. Power-to-dB dönüşümü
  3. Mel ölçeğine göre frekans bantlarının yeniden örneklenmesi
- **Çıktı Boyutu:** (128, 174) veya transfer öğrenme için (224, 224, 3)

#### Avantajlar
- İnsan işitme sistemine uygun frekans çözünürlüğü
- Formantlar ve harmoniklerin görsel doku olarak korunması
- Transient (ani değişimli) seslerde yüksek performans
- CNN mimarileri için ideal 2D görsel temsil

#### Kod Örneği
```python
import numpy as np
import librosa

def extract_melspectrogram(file_path, n_mels=128, max_len=174):
    """Mel-Spektrogram özelliklerini ses dosyasından çıkarır."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256,
        fmin=20,
        fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Sabit boyuta getirme
    if mel_spec_db.shape[1] < max_len:
        pad_w = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,pad_w)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]

    return mel_spec_db
```

### 2. MFCC (Mel-Frequency Cepstral Coefficients)

MFCC, ses sinyalinin spektral zarfının kompakt bir özetini sağlar ve konuşma tanıma sistemlerinde yaygın olarak kullanılır.

#### Teknik Detaylar
- **MFCC Katsayısı:** 40
- **Örnekleme Frekansı:** 44.1 kHz
- **Pencere Fonksiyonu:** Kaiser_fast
- **Süre:** 4 saniye
- **İşlem Adımları:**
  1. Mel-Spektrogram hesaplanması
  2. DCT (Discrete Cosine Transform) uygulanması
  3. Delta (Δ) ve Delta-Delta (ΔΔ) türevlerinin hesaplanması (opsiyonel)
- **Çıktı Boyutu:** (40, 174)

#### Avantajlar
- Kompakt temsil, düşük boyutluluk
- Gürültüye karşı dayanıklılık
- Daha az hesaplama gereksinimi (%15 daha hızlı eğitim)
- Kararlı ve stabil precision değerleri

#### Kod Örneği
```python
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """MFCC özelliklerini ses dosyasından çıkarır."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Sabit boyuta getirme
    if mfcc.shape[1] < max_len:
        pad_w = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_w)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
```

### 3. YAMNet Embeddings

YAMNet (Yet Another Mobile Network), Google tarafından AudioSet veri seti üzerinde eğitilmiş bir ses sınıflandırma modelidir.

- **Embedding Boyutu:** 1024 boyutlu vektör
- **Model Kaynağı:** TensorFlow Hub (https://tfhub.dev/google/yamnet/1)
- **Giriş Özellikleri:** 16 kHz mono ses
- **Avantaj:** Önceden eğitilmiş ses temsilleri, transfer öğrenme için ideal

### 4. PANNs Embeddings (Pre-trained Audio Neural Networks)

PANNs, AudioSet veri seti üzerinde eğitilmiş CNN14 mimarisine sahip bir ses özellik çıkarıcıdır.

- **Embedding Boyutu:** ~2048 boyutlu vektör (katmana göre değişir)
- **Mimari:** CNN14
- **Avantaj:** Yüksek performanslı özellik temsilleri, çeşitli sınıflandırıcılarla uyumlu

---

## Model Mimarileri

Proje kapsamında **25+ farklı model mimarisi** ve **19 eğitim scripti** geliştirilmiştir.

### A. Özel CNN Mimarileri

#### 1. Mel-Spektrogram CNN (`train_melspectrogram_cnn.py`)

**Mimari Yapısı:**
```
Input: (128, 174, 1) - Mel-Spektrogram

Block 1:
├─ Conv2D(32, 3×3, activation='relu')
├─ Conv2D(32, 3×3, activation='relu')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)
└─ Dropout(0.25)

Block 2:
├─ Conv2D(64, 3×3, activation='relu')
├─ Conv2D(64, 3×3, activation='relu')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)
└─ Dropout(0.25)

Block 3:
├─ Conv2D(128, 3×3, activation='relu')
├─ Conv2D(128, 3×3, activation='relu')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)
└─ Dropout(0.25)

Block 4:
├─ Conv2D(256, 3×3, activation='relu')
├─ Conv2D(256, 3×3, activation='relu')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)
└─ Dropout(0.25)

Classifier:
├─ Flatten()
├─ Dense(512, activation='relu')
├─ Dropout(0.5)
├─ Dense(256, activation='relu')
├─ Dropout(0.5)
└─ Dense(10, activation='softmax')
```

**Eğitim Parametreleri:**
- Epoch: 50
- Batch Size: 32
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)

**Performans:** %94.2 test doğruluğu

#### 2. MFCC CNN (`train_mfcc_cnn.py`)

**Mimari Yapısı:**
```
Input: (40, 174, 1) - MFCC

Block 1:
├─ Conv2D(32, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.25)

Block 2:
├─ Conv2D(64, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.25)

Block 3:
├─ Conv2D(128, 3×3) + BatchNorm + MaxPool(2×2) + Dropout(0.25)

Classifier:
├─ Flatten() → Dense(256) → Dense(128) → Softmax(10)
```

**Performans:** %93.1 test doğruluğu, %15 daha hızlı eğitim

### B. Transfer Öğrenme Modelleri

#### 3. ResNet50 + Mel-Spektrogram (`train_mel_resnet50.py`)

- **Temel Model:** ResNet50 (ImageNet ön-eğitimli)
- **Giriş:** Log-mel spektrogramlar (224×224×3)
- **Örnekleme Frekansı:** 32 kHz
- **Mel Konfigürasyonu:** 128 mel band, FFT=1024, hop=256
- **Performans:** %88.0 test doğruluğu
- **Özellik:** Dondurulmuş temel + eğitilebilir üst katmanlar

#### 4. MobileNetV2 + Mel-Spektrogram (`train_mel_mobilenetv2.py`)

- **Temel Model:** MobileNetV2 (hafif mimari)
- **Giriş:** Log-mel spektrogramlar (224×224×3)
- **Avantaj:** Düşük hesaplama maliyeti, mobil dağıtıma uygun
- **Mimari:** Global average pooling → Dense katmanlar → Softmax

#### 5. Xception + Mel-Spektrogram (`train_melspectogram_xception.py`)

- **Temel Model:** Xception (depthwise separable convolutions)
- **Özellikler:**
  - GPU algılama (Apple Metal, NVIDIA)
  - Dinamik katman dondurma
  - Gelişmiş metrikler: Balanced accuracy, Cohen's kappa, Matthews correlation coefficient
- **Desteklenen Alternatifler:** VGG16/19, ResNet varyantları, Inception, DenseNet, EfficientNet

#### 6. VGG16 + MFCC (`train_mfcc_vgg16.py`)

- **Temel Model:** VGG16 (ve komut satırı ile alternatifler)
- **Desteklenen Modeller:**
  - VGG (16, 19)
  - ResNet (50, 101, 152)
  - Inception varyantları
  - MobileNet, DenseNet, EfficientNet, Xception
- **Kullanım:** `python train_mfcc_vgg16.py VGG16`

#### 7. DenseNet121 + MFCC (`train_mfcc_densenet121.py`)

- **Temel Model:** DenseNet121 (yoğun bağlantılar)
- **GPU Desteği:** Apple Silicon (M1/M2/M3), NVIDIA GPU algılama
- **Gelişmiş Metrikler:**
  - Balanced accuracy
  - Cohen's kappa score
  - Matthews correlation coefficient
  - Top-k accuracy

### C. Önceden Eğitilmiş Ses Modellerinden Transfer Öğrenme

#### 8. YAMNet Transfer Learning (`train_yamnet_transfer.py`)

- **Temel Model:** YAMNet (Google'ın önceden eğitilmiş ses modeli)
- **Embedding Boyutu:** 1024 boyutlu
- **Model URL:** https://tfhub.dev/google/yamnet/1
- **Giriş Özellikleri:** 16 kHz mono ses
- **Mimari:** Dondurulmuş YAMNet + özel sınıflandırıcı
- **Sınıflandırıcı Katmanları:**
  ```
  Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Softmax(10)
  ```
- **Toplam Parametre:** 2,070,944 (7.90 MB)

#### 9. Geliştirilmiş YAMNet (`train_yamnet_improved.py`)

Birden fazla mimari varyasyonunu destekler:

- **shallow:** Hızlı temel model
- **medium:** Dengeli karmaşıklık
- **deep:** Maksimum kapasite
- **residual:** Residual bağlantılar
- **ensemble:** Çok dallı mimari

**Konfigürasyon Seçenekleri:**
- Dropout Oranı: 0.3-0.5 ayarlanabilir
- Learning Rate: 1e-4, 5e-4, 1e-3 seçenekleri
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#### 10. PANNs CNN14 (`train_panns.py`) - EN BÜYÜK SCRIPT (1,003 satır)

- **Temel Model:** PANNs CNN14 mimarisi
- **Özellik Çıkarıcı:** Dondurulmuş CNN14 embeddings (~2048-dim)
- **Desteklenen Sınıflandırıcılar:**
  - Logistic Regression
  - MLP (Multi-layer Perceptron)
  - LightGBM (opsiyonel)
- **Veri Artırma:** SpecAugment (zaman/frekans maskeleme)
- **Ön İşleme Özellikleri:**
  - Ses yükleme ve yeniden örnekleme
  - Önbelleklenmiş embedding hesaplama
  - Ortalama/standart sapma normalizasyonu
  - Opsiyonel veri artırma
- **Kapsamlı Değerlendirme:**
  - Accuracy, precision, recall, F1-score
  - Confusion matrix ve classification report
  - Cross-validation seçeneği
- **Performans:** %94.7 test doğruluğu

### Model Mimarileri Özet Tablosu

| Model Tipi | Adet | Örnekler |
|-----------|------|----------|
| Özel CNN | 2 | Mel-Spektrogram CNN, MFCC CNN |
| ResNet Ailesi | 3+ | ResNet50, ResNet101, ResNet152 |
| MobileNet Ailesi | 1+ | MobileNetV2 |
| VGG Ailesi | 2+ | VGG16, VGG19 |
| Inception Ailesi | 2+ | InceptionV3, InceptionResNetV2 |
| DenseNet Ailesi | 3+ | DenseNet121, DenseNet169, DenseNet201 |
| EfficientNet Ailesi | 4+ | EfficientNetB0-B3 |
| Xception | 1 | Mel özellikleriyle Xception |
| YAMNet | 3 | Transfer + Geliştirilmiş (birden fazla varyant) |
| PANNs | 1 | Çeşitli sınıflandırıcılarla CNN14 |
| **Toplam Benzersiz Model** | **25+** | - |

---

## Performans Sonuçları

### En İyi Performans Gösteren Modeller

| Model | Özellik | Doğruluk | Temel Avantaj |
|-------|---------|----------|---------------|
| **PANNs CNN14 + MLP** | PANNs Embeddings | **94.7%** | En yüksek genel performans |
| **Mel-Spektrogram CNN** | Mel-Spektrogram | **94.2%** | Transient seslerde mükemmel |
| **MFCC CNN** | MFCC (40 katsayı) | **93.1%** | %15 daha hızlı, gürültüye dayanıklı |
| **ResNet50 + MFCC** | MFCC | **88.0%** | Transfer öğrenme temeli |

### Sınıf Bazlı Performans Örnekleri (ResNet50 + MFCC)

- **air_conditioner:** %94 precision
- **gun_shot:** %97.3 recall
- **drilling:** %92 F1-score
- **dog_bark:** %85.5 recall

### Mel-Spektrogram CNN Detaylı Analizi

**En Yüksek Doğruluk:**
- air_conditioner: %98
- drilling: %94

**Daha Düşük Doğruluk:**
- children_playing: Transient yapı nedeniyle
- dog_bark: Transient yapı nedeniyle

**Genel Metrikler:**
- Test Doğruluğu: %94.2
- Yüksek recall değerleri
- Karmaşık ses dokularında başarılı

### MFCC CNN Detaylı Analizi

**En Yüksek Doğruluk:**
- air_conditioner: %97.5
- children_playing: %97.5

**En Düşük Doğruluk:**
- dog_bark: %89

**Genel Metrikler:**
- Test Doğruluğu: %93.1
- Daha stabil precision değerleri
- %15 daha hızlı eğitim süresi
- Düşük gürültülü ortamlarda üstün performans

---

## Kurulum

### Gereksinimler

Projede 3 farklı requirements dosyası bulunmaktadır:

#### 1. Temel Gereksinimler (`requirements.txt`)
```bash
pip install -r requirements.txt
```

**İçerik:**
```
numpy
librosa          # Ses işleme
scikit-learn     # ML araçları
tensorflow       # Derin öğrenme framework
matplotlib       # Görselleştirme
seaborn          # İstatistiksel görselleştirme
pandas           # Veri manipülasyonu
scipy            # Bilimsel hesaplama
resampy          # Ses yeniden örnekleme
```

#### 2. YAMNet İçin Gereksinimler (`requirements_yamnet.txt`)
```bash
pip install -r requirements_yamnet.txt
```

**İçerik:**
```
tensorflow>=2.12.0
tensorflow-hub>=0.13.0
numpy>=1.23.0
librosa>=0.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

#### 3. XAI (Açıklanabilir Yapay Zeka) İçin Gereksinimler (`requirements_xai.txt`)
```bash
pip install -r requirements_xai.txt
```

**İçerik:**
```
tensorflow>=2.12.0
tensorflow-hub>=0.13.0
numpy>=1.23.0
librosa>=0.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
lime>=0.2.0              # Local Interpretable Model-agnostic Explanations
shap>=0.42.0             # SHapley Additive exPlanations
```

### Veri Setinin Hazırlanması

1. UrbanSound8K veri setini indirin
2. Ses dosyalarını aşağıdaki yapıya göre organize edin:

```
dataset/
├── air_conditioner/
├── car_horn/
├── children_playing/
├── dog_bark/
├── drilling/
├── engine_idling/
├── gun_shot/
├── jackhammer/
├── siren/
└── street_music/
```

---

## Kullanım

### 1. Özel CNN Modelleri ile Eğitim

#### Mel-Spektrogram CNN
```bash
python train_melspectrogram_cnn.py
```

**Çıktılar:**
- `results/melspectrogram/model_melspectrogram.h5` - Eğitilmiş model
- `results/melspectrogram/results.json` - Metrikler ve sınıflandırma raporu
- `results/melspectrogram/training_history.png` - Eğitim grafiği
- `results/melspectrogram/confusion_matrix.png` - Confusion matrix
- `results/melspectrogram/confusion_matrix_normalized.png` - Normalize confusion matrix
- `results/melspectrogram/class_accuracy.png` - Sınıf bazlı doğruluk grafiği

#### MFCC CNN
```bash
python train_mfcc_cnn.py
```

**Çıktılar:**
- `results/mfcc/model_mfcc.h5`
- `results/mfcc/results.json`
- `results/mfcc/training_history.png`
- `results/mfcc/confusion_matrix.png`
- Diğer görselleştirme dosyaları

### 2. Transfer Öğrenme Modelleri

#### ResNet50 + Mel-Spektrogram
```bash
python train_mel_resnet50.py
```

#### MobileNetV2 + Mel-Spektrogram
```bash
python train_mel_mobilenetv2.py
```

#### VGG16 + MFCC (Model seçimi ile)
```bash
# VGG16 kullanarak
python train_mfcc_vgg16.py VGG16

# ResNet50 kullanarak
python train_mfcc_vgg16.py ResNet50

# DenseNet121 kullanarak
python train_mfcc_vgg16.py DenseNet121
```

#### DenseNet121 + MFCC
```bash
python train_mfcc_densenet121.py
```

#### Xception + Mel-Spektrogram
```bash
python train_melspectogram_xception.py
```

### 3. Önceden Eğitilmiş Ses Modelleri

#### YAMNet Transfer Learning
```bash
python train_yamnet_transfer.py
```

#### Geliştirilmiş YAMNet (Birden fazla varyant)
```bash
python train_yamnet_improved.py
```

**Konfigürasyon seçenekleri (script içinde):**
- `CONFIG['architecture']`: 'shallow', 'medium', 'deep', 'residual', 'ensemble'
- `CONFIG['dropout_rate']`: 0.3, 0.4, 0.5
- `CONFIG['learning_rate']`: 1e-4, 5e-4, 1e-3

#### PANNs CNN14
```bash
python train_panns.py
```

### 4. Analiz ve Görselleştirme Araçları

#### Ses Özelliklerini Görselleştirme
```bash
python visualize_audio_features.py
```

#### t-SNE Görselleştirmesi (Mel-Spektrogram)
```bash
python tsne_mel.py
```

#### YAMNet Model Karşılaştırması
```bash
python compare_yamnet_models.py
```

#### MFCC Örnek Tahminler
```bash
python create_mfcc_sample_predictions.py
```

### 5. Açıklanabilir Yapay Zeka (XAI)

#### XAI Açıklayıcı
```bash
python xai_explainer.py
```

**Özellikler:**
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Özellik önem analizi
- Zamansal katkı haritaları

#### XAI Örnek Kullanım
```bash
python xai_example.py
```

#### XAI Hızlı Test
```bash
python xai_quick_test.py
```

---

## Proje Yapısı

```
sound-classification/
│
├── README.md                              # Proje dokümantasyonu
├── requirements.txt                       # Temel bağımlılıklar
├── requirements_yamnet.txt                # YAMNet bağımlılıkları
├── requirements_xai.txt                   # XAI bağımlılıkları
│
├── dataset/                               # Veri seti dizini
│   ├── air_conditioner/                   # 1,000 örnek
│   ├── car_horn/                          # 429 örnek
│   ├── children_playing/                  # 1,000 örnek
│   ├── dog_bark/                          # 1,000 örnek
│   ├── drilling/                          # 1,000 örnek
│   ├── engine_idling/                     # 1,000 örnek
│   ├── gun_shot/                          # 374 örnek
│   ├── jackhammer/                        # 1,000 örnek
│   ├── siren/                             # 929 örnek
│   └── street_music/                      # 1,000 örnek
│
├── archive/                               # UrbanSound8K orijinal yapısı
│
├── results/                               # Eğitim sonuçları ve çıktılar
│   ├── melspectrogram/                    # Mel-Spec CNN sonuçları
│   ├── melspectrogram_cnn/                # Alternatif Mel-Spec CNN varyantı
│   ├── melspectrogram_resnet50/           # ResNet50 + Mel-Spec sonuçları
│   ├── melspectrogram_xception/           # Xception + Mel-Spec sonuçları
│   ├── mfcc/                              # MFCC CNN sonuçları
│   ├── mfcc_vgg16/                        # VGG16 + MFCC sonuçları
│   ├── mfcc_densenet121/                  # DenseNet121 + MFCC sonuçları
│   ├── mfcc_mobilenetv2/                  # MobileNetV2 + MFCC sonuçları
│   ├── mfcc_resnet50/                     # ResNet50 + MFCC sonuçları (88.0% acc)
│   ├── mfcc_efficientnetb3/               # EfficientNetB3 + MFCC sonuçları
│   ├── panns_cnn14/                       # PANNs CNN14 sonuçları (94.7% acc)
│   ├── yamnet_transfer/                   # YAMNet transfer learning sonuçları
│   ├── yamnet_improved_deep/              # Geliştirilmiş YAMNet (deep varyant)
│   ├── yamnet_improved_residual/          # Geliştirilmiş YAMNet (residual varyant)
│   ├── yamnet_improved_shallow/           # Geliştirilmiş YAMNet (shallow varyant)
│   ├── yamnet_improved_ensemble/          # Geliştirilmiş YAMNet (ensemble varyant)
│   ├── tsne_mel/                          # t-SNE görselleştirme sonuçları
│   ├── feature_visualizations/            # Özellik görselleştirme grafikleri
│   ├── methodology/                       # Metodoloji dokümantasyonu
│   └── xai/                               # XAI açıklama sonuçları
│
├── Eğitim Scriptleri (19 dosya, 6,191 satır kod)
│   ├── train_melspectrogram_cnn.py        # (323 satır) Mel-Spec CNN
│   ├── train_mfcc_cnn.py                  # (312 satır) MFCC CNN
│   ├── train_mel_resnet50.py              # (339 satır) ResNet50 + Mel
│   ├── train_mel_mobilenetv2.py           # (356 satır) MobileNetV2 + Mel
│   ├── train_mfcc_vgg16.py                # (383 satır) VGG16 + MFCC
│   ├── train_mfcc_densenet121.py          # (448 satır) DenseNet121 + MFCC
│   ├── train_melspectogram_xception.py    # (477 satır) Xception + Mel
│   ├── train_yamnet_transfer.py           # (405 satır) YAMNet transfer
│   ├── train_yamnet_improved.py           # (666 satır) Geliştirilmiş YAMNet
│   └── train_panns.py                     # (1,003 satır) PANNs CNN14
│
├── Analiz ve Görselleştirme Scriptleri
│   ├── compare_yamnet_models.py           # (201 satır) YAMNet model karşılaştırma
│   ├── visualize_audio_features.py        # (69 satır) Ses özelliği görselleştirme
│   ├── tsne_mel.py                        # (165 satır) t-SNE görselleştirme
│   ├── create_mfcc_sample_predictions.py  # (117 satır) MFCC örnek tahminler
│   └── generate_methodology_assets.py     # Metodoloji varlık üretimi
│
├── XAI (Açıklanabilir Yapay Zeka) Scriptleri
│   ├── xai_explainer.py                   # (504 satır) XAI açıklayıcı
│   ├── xai_example.py                     # (254 satır) XAI örnek kullanım
│   └── xai_quick_test.py                  # XAI hızlı test
│
└── Yardımcı Scriptler
    └── move_air_conditioner.py            # Veri seti organizasyon yardımcısı
```

### Her Sonuç Dizininde Tipik Olarak Oluşturulan Dosyalar

- **model_*.h5** - Eğitilmiş Keras modeli
- **results.json** - Metrikler ve sınıflandırma raporu
- **training_history.png** - Loss/accuracy eğrileri
- **confusion_matrix.png** - Ham confusion matrix
- **confusion_matrix_normalized.png** - Normalize confusion matrix
- **class_accuracy.png** - Sınıf bazlı doğruluk çubuk grafiği
- **learning_rate.png** - Öğrenme oranı programı
- **sample_predictions.png** - Tahminlerle birlikte görsel örnekler

---

## Özel Özellikler ve Teknikler

### A. Veri Artırma

**Planlanan stratejiler** (metodoloji dokümante edilmiştir):
- **SNR hedefleri:** +5 dB, +0 dB, -5 dB, -10 dB
- **Zaman genişletme:** 0.9-1.1 oranları
- **Pitch shifting:** -2 ile +2 yarım ton arası
- **SpecAugment:** Zaman/frekans maskeleme (32 frame, 8 mel band)

### B. Düzenlileştirme Teknikleri

- **Dropout:** Katman başına 0.25-0.5
- **Batch Normalization:** Konvolüsyonel katmanlardan sonra
- **L2 Regularization:** Bazı modellerde uygulanmıştır
- **Early Stopping:** patience=4-15 epoch
- **Learning Rate Reduction:** factor=0.5, patience=5

### C. GPU Optimizasyonu

- Apple Silicon (Metal GPU) algılama ve optimizasyonu
- NVIDIA GPU bellek büyümesi konfigürasyonu
- Karma hassasiyet eğitimi (Apple Silicon için opsiyonel)
- Dağıtık eğitim desteği (altyapı mevcut)

### D. Açıklanabilir Yapay Zeka (XAI)

- **LIME:** Local Interpretable Model-agnostic Explanations
- **SHAP:** SHapley Additive exPlanations
- **Attention Visualization:** Attention tabanlı modeller için
- **Özellik Önem Analizi:** Model başına özellik katkısı
- **Zamansal Katkı Haritaları:** Zaman serisi önemi

### E. Gelişmiş Metrikler

- **Balanced Accuracy:** Sınıf dengesizliğini ele alır
- **Cohen's Kappa:** Şans uyumunu düzeltir
- **Matthews Correlation:** Tek skorlu sınıflandırıcı kalitesi
- **Top-k Accuracy:** Çok sınıflı problemler için faydalı

---

## Gelecek Çalışmalar

### A. Tıbbi Uygulama: Gastroenterolojik Ses Analizi

Bu çalışmanın altyapısı, **gastroenteroloji** alanına uyarlanarak bağırsak seslerinden hastalık tahmini yapılması için genişletilecektir.

#### Hedef Hastalıklar
- Irritabl bağırsak sendromu (IBS)
- Kabızlık
- İshal
- Bağırsak tıkanıklığı

#### Planlanan Adımlar

1. **Veri Toplama**
   - Yüksek hassasiyetli mikrofonlarla bağırsak ses kayıtları
   - Normal ve patolojik bağırsak seslerinin etiketlenmesi
   - Klinik ortamlarda veri toplama protokolleri

2. **Özellik Çıkarımı**
   - MFCC ve Mel-Spektrogram temsilleri
   - Harmonik-Perküsif kaynak ayrımı (HPSS)
   - Temporal özellik analizi

3. **Model Eğitimi**
   - CNN/CRNN mimarileriyle sınıflandırma
   - Transfer öğrenme teknikleri
   - Ensemble yöntemleri

4. **Klinik Entegrasyon**
   - Taşınabilir erken teşhis destek sistemi
   - Mobil uygulama geliştirme
   - Nokta-of-care tanı araçları

#### Teknolojik Vizyon

- **Tıbbi Ses Analizi:** Bağırsak, akciğer, kalp seslerinden otomatik anormallik tespiti
- **Mobil Uygulama:** Akıllı telefon mikrofonlarıyla anlık analiz
- **Klinik Yardımcı:** Doktorlara akustik verilerle desteklenmiş karar desteği

### B. Model İyileştirmeleri

1. **Hibrit Modeller**
   - Mel + MFCC kombinasyonu ile daha dengeli performans
   - Çoklu özellik füzyonu teknikleri
   - Attention mekanizmaları ile özellik ağırlıklandırma

2. **Temporal Modeller**
   - GRU veya LSTM katmanları ekleme
   - Transformer mimarileri (Audio Transformers)
   - Temporal Convolutional Networks (TCN)

3. **Gelişmiş Veri Artırma**
   - Kapsamlı noise injection
   - Time stretching ve pitch shifting
   - Mixup ve SpecAugment teknikleri
   - Generative adversarial networks (GAN) ile veri üretimi

4. **Transfer Öğrenme**
   - Büyük ses modellerinden (YAMNet, AudioSet, Wav2Vec 2.0) fine-tuning
   - Domain adaptation teknikleri
   - Few-shot learning yaklaşımları

5. **Ensemble Yöntemleri**
   - Birden fazla modelin tahminlerini birleştirme
   - Stacking ve blending teknikleri
   - Model çeşitliliği optimizasyonu

### C. Dağıtım ve Uygulama

1. **Edge Deployment**
   - Mobil cihazlara dağıtım (TensorFlow Lite, ONNX)
   - Model hafifletme ve kuantizasyon
   - Gerçek zamanlı çıkarım optimizasyonu

2. **Gerçek Zamanlı İşleme**
   - Streaming inference
   - Düşük gecikme süresi optimizasyonları
   - Online learning mekanizmaları

3. **Kullanıcı Arayüzü**
   - Web tabanlı demo uygulaması
   - Mobil uygulama (iOS/Android)
   - API servisleri

---

## Değerlendirme Metrikleri

Proje kapsamında kullanılan değerlendirme metrikleri:

- **Accuracy:** Genel doğruluk
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Precision ve recall'un harmonik ortalaması
- **Confusion Matrix:** Sınıf bazlı detaylı performans
- **Balanced Accuracy:** Sınıflar arası recall ortalaması
- **Cohen's Kappa:** Şans düzeltmeli inter-rater uyumu
- **Matthews Correlation Coefficient:** Tahminler ve etiketler arası korelasyon
- **Top-k Accuracy:** Tahmin top-k tahminler içinde

---

## Referanslar

### Veri Seti
- **UrbanSound8K:** J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

### Framework ve Kütüphaneler
- **TensorFlow/Keras:** Derin öğrenme framework
- **Librosa:** Ses işleme kütüphanesi
- **Scikit-learn:** Makine öğrenmesi araçları
- **TensorFlow Hub:** Önceden eğitilmiş modeller (YAMNet)
- **PANNs:** Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"

### Özellik Çıkarım Yöntemleri
- **Mel-Spektrogram:** Mel-Frequency Analysis
- **MFCC:** Davis & Mermelstein, "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", 1980

### XAI Araçları
- **LIME:** Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier", 2016
- **SHAP:** Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", 2017

---

## İletişim

Proje hakkında sorularınız için ekip üyeleriyle iletişime geçebilirsiniz:

**Proje Ekibi:**
- Furkan Yılmaz
- Aleyna Taşdemir
- Edanur Arslan
- Talip Kurt

**Danışman:**
- Doç. Dr. Fatih Ekinci

---

## Lisans

Bu proje akademik araştırma amaçlıdır.

---

**Not:** Bu proje, ses tabanlı sinyal işleme tekniklerinin sağlık teknolojileriyle birleştiği yeni bir araştırma hattı açmaktadır. Çevresel ses sınıflandırmadan tıbbi akustik analize geçiş, yapay zekânın klinik karar destek sistemlerinde kullanılabilirliğini güçlendirecek ve gelecekte gastroenteroloji alanında erken teşhis odaklı bir inovasyon temelini oluşturacaktır.

**Proje İstatistikleri:**
- **Toplam Kod Satırı:** 6,191+
- **Eğitim Scripti:** 19
- **Benzersiz Model Mimarisi:** 25+
- **Veri Seti Boyutu:** 8,732 ses dosyası
- **En Yüksek Doğruluk:** %94.7 (PANNs CNN14)
- **Desteklenen Özellik Temsilleri:** 4 (Mel-Spektrogram, MFCC, YAMNet, PANNs)
