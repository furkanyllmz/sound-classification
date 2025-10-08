# Derin Öğrenme Tabanlı Ses Sınıflandırma ve Gelecek Uygulamaları

**Proje Ekibi:** Furkan Yılmaz, Aleyna Taşdemir, Edanur Arslan, Talip Kurt
**Danışman:** Doç. Dr. FATİH EKİNCİ
**Tarih:** Ekim 2025

## 📋 Özet

Bu proje, ses sinyallerinin yapay zeka tabanlı sınıflandırılmasına yönelik bir çalışmadır. Ses verilerinden **Mel-Spektrogram** ve **MFCC** temsilleri çıkarılmış, bu temsiller **Konvolüsyonel Sinir Ağları (CNN)** ile eğitilerek farklı ses sınıflarının tanınması sağlanmıştır.

### Sınıflandırılan Ses Türleri
- 🔊 Klima sesi (air_conditioner)
- 🚗 Araba kornası (car_horn)
- 👶 Çocuk sesleri (children_playing)
- 🐕 Köpek havlaması (dog_bark)
- 🔧 Matkap sesi (drilling)

## 🎯 Ana Bulgular

- **Mel-Spektrogram CNN:** %94.2 doğruluk - Transient (ani değişimli) seslerde üstün performans
- **MFCC-CNN:** %93.1 doğruluk - Gürültüye dayanıklı ortamlarda daha stabil sonuçlar

| Model | Doğruluk | Avantajlı Olduğu Durum |
|-------|----------|------------------------|
| Mel-Spektrogram CNN | %94.2 | Yüksek frekanslı, karmaşık transient sesler |
| MFCC-CNN | %93.1 | Düşük gürültülü, kararlı sinyaller |

## 🔬 Teknik Yaklaşım

### 1. Fourier Analizi
- **DFT (Discrete Fourier Transform):** Zaman domeninden frekans domenine dönüşüm
- **STFT (Short-Time Fourier Transform):** Zaman-frekans analizi
- **Pencereleme:** Hamming/Hann pencereleri ile frekans sızıntısı önleme

### 2. Mel Spektrogram
- İnsan kulağının algısal özelliklerini modelleyen Mel ölçeği
- 128 mel bandı kullanımı
- Yüksek frekans çözünürlüğü
- CNN için uygun 2B görsel temsil

**Avantajlar:**
- İnsan işitme sistemine yakın frekans çözünürlüğü
- Formantlar ve harmoniklerin görsel doku olarak korunması
- Transient seslerde yüksek performans

### 3. MFCC (Mel-Frequency Cepstral Coefficients)
- 40 MFCC katsayısı çıkarımı
- DCT ile spektral zarfın kompakt özeti
- Delta (Δ) ve Delta-Delta (ΔΔ) türevleri

**Avantajlar:**
- Kompakt temsil, düşük boyut
- Gürültüye dayanıklılık
- Daha az hesaplama gereksinimi

## 🏗️ Model Mimarisi

### CNN Yapısı
```
1. Blok: 2 × Conv2D(32, 3×3) + BatchNorm + MaxPooling(2,2) + Dropout
2. Blok: 2 × Conv2D(64, 3×3) + BatchNorm + MaxPooling(2,2) + Dropout
3. Blok: 2 × Conv2D(128, 3×3) + BatchNorm + MaxPooling(2,2) + Dropout
Flatten → Dense(512) → Dense(256) → Softmax(5 sınıf)
```

### Eğitim Parametreleri
- **Epoch:** 20
- **Batch Size:** 32
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Callbacks:** EarlyStopping(patience=4) + ReduceLROnPlateau
- **Donanım:** Apple M1 GPU (~1 saat eğitim süresi)

## 📊 Veri Seti

**UrbanSound8K** veri setinin alt kümesi kullanılmıştır:
- **Ses Uzunluğu:** 4 saniye (sabit)
- **Örnekleme Frekansı:** 44.1 kHz
- **Kanal:** Mono
- **Sınıf Sayısı:** 5
- **Normalizasyon:** Uygulandı

## 💻 Kurulum ve Kullanım

### Gereksinimler
```bash
pip install numpy librosa tensorflow keras matplotlib
```

### Mel-Spektrogram Özellik Çıkarımı
```python
import numpy as np
import librosa

def extract_melspectrogram(file_path, n_mels=128, max_len=174):
    """Mel-Spektrogram özelliklerini ses dosyasından çıkarır."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_len:
        pad_w = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,pad_w)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]
    return mel_spec_db
```

### MFCC Özellik Çıkarımı
```python
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    """MFCC özelliklerini ses dosyasından çıkarır."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_w = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_w)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc
```

## 📈 Performans Analizi

### Mel-Spektrogram CNN
- ✅ En yüksek doğruluk: air_conditioner (%98), drilling (%94)
- ⚠️ Daha düşük doğruluk: children_playing ve dog_bark (transient yapı)
- 📊 Genel test doğruluğu: %94.2
- 🎯 Yüksek recall değerleri

### MFCC-CNN
- ✅ En yüksek doğruluk: air_conditioner (%97.5), children_playing (%97.5)
- ⚠️ En düşük doğruluk: dog_bark (%89)
- 📊 Genel test doğruluğu: %93.1
- 🎯 Daha stabil precision değerleri
- ⚡ %15 daha hızlı eğitim süresi

## 🔮 Gelecek Çalışmalar

### Tıbbi Uygulama: Bağırsak Seslerinden Hastalık Tahmini

Bu çalışmanın altyapısı, **gastroenteroloji** alanına uyarlanarak bağırsak seslerinden hastalık tahmini yapılması için genişletilecektir.

#### Hedef Hastalıklar
- Irritabl bağırsak sendromu
- Kabızlık
- İshal
- Tıkanıklık

#### Planlanan Adımlar
1. 🎤 Yüksek hassasiyetli mikrofonlarla ses kayıtları toplama
2. 🔊 MFCC ve Mel-Spektrogram temsilleriyle özellik çıkarımı
3. 🧠 CNN/CRNN mimarileriyle sınıflandırma
4. 🏥 Normal ve patolojik bağırsak seslerinin etiketlenmesi
5. 📱 Taşınabilir erken teşhis destek sistemi geliştirme

### Teknolojik Vizyon
- **Tıbbi Ses Analizi:** Bağırsak, akciğer, kalp seslerinden otomatik anormallik tespiti
- **Mobil Uygulama:** Akıllı telefon mikrofonlarıyla anlık analiz
- **Klinik Yardımcı:** Doktorlara akustik verilerle desteklenmiş karar desteği

## 🚀 Gelecekte Yapılabilecek İyileştirmeler

1. **Hibrit Model:** Mel + MFCC birleşimi ile daha dengeli performans
2. **Temporal Modeller:** GRU veya Transformer katmanları ekleme
3. **Veri Artırma:** Noise injection, time stretching, pitch shifting
4. **Transfer Learning:** Büyük ses modellerinden (YAMNet, AudioSet) fine-tuning
5. **Ensemble Yaklaşımı:** Birden fazla modelin tahminlerini birleştirme

## 📚 Referanslar

- **Veri Seti:** UrbanSound8K
- **Framework:** TensorFlow/Keras
- **Ses İşleme:** Librosa
- **Model Mimarisi:** Konvolüsyonel Sinir Ağları (CNN)

## 📞 İletişim

Proje hakkında sorularınız için ekip üyeleriyle iletişime geçebilirsiniz.

---

**Not:** Bu proje, ses tabanlı sinyal işleme tekniklerinin sağlık teknolojileriyle birleştiği yeni bir araştırma hattı açmaktadır. Çevresel ses sınıflandırmadan tıbbi akustik analize geçiş, yapay zekânın klinik karar destek sistemlerinde kullanılabilirliğini güçlendirecek ve gelecekte gastroenteroloji alanında erken teşhis odaklı bir inovasyon temelini oluşturacaktır.
