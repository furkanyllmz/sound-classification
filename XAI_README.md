# XAI (Explainable AI) - Ses Sınıflandırma için Açıklanabilir Yapay Zeka

## 🎯 Amaç
Derin öğrenme modellerinin ses sınıflandırmada **NASIL** ve **NEDEN** karar verdiğini görselleştirmek ve anlamak.

---

## 📊 Uygulanan XAI Teknikleri

### 1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- **Dosya:** `xai_gradcam_analysis.py`
- **Ne Yapar:** Modelin hangi bölgelere odaklandığını gösterir
- **Formül:** 
  ```
  L_Grad-CAM = ReLU(Σ_k α_k × A_k)
  α_k = (1/Z) × Σ_i Σ_j (∂y^c / ∂A_k_ij)
  ```
- **Çıktı:** Heatmap görselleri

### 2. **LIME (Local Interpretable Model-agnostic Explanations)**
- **Dosya:** `xai_lime_analysis.py`
- **Ne Yapar:** Model-agnostik lokal açıklamalar
- **Formül:**
  ```
  ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)
  ```
- **Çıktı:** Segment bazlı önemlilik görselleri

### 3. **Feature Importance (Occlusion, Gradients, Integrated Gradients)**
- **Dosya:** `xai_feature_importance.py`
- **Ne Yapar:** Hangi MFCC katsayılarının önemli olduğunu gösterir
- **Yöntemler:**
  - **Occlusion Sensitivity:** Bölgeleri sıfırlayarak önemini test eder
  - **Gradient-based:** `|∂P(c)/∂x|` ile importance hesaplar
  - **Integrated Gradients:** Path integral ile attribution hesaplar

---

## 📦 Gerekli Kütüphaneler

### Yükleme
```bash
pip install opencv-python
pip install lime
pip install scikit-image
```

Diğer kütüphaneler zaten yüklü:
- tensorflow
- numpy
- matplotlib
- seaborn
- librosa
- scikit-learn

---

## 🚀 Kullanım

### 1. Grad-CAM Analizi
```python
python xai_gradcam_analysis.py
```

**Özelleştirme:**
```python
# Script içinde düzenleyin:
class_names = ['sınıf1', 'sınıf2', ...]  # Sınıf isimleri
MODEL_PATH = 'results/mfcc/mfcc_cnn_model.h5'  # Model yolu
sample_audio = 'dataset/dog_bark/sample.wav'  # Test dosyası

# Analiz
analyze_single_prediction(MODEL_PATH, sample_audio, class_names)
analyze_multiple_layers(MODEL_PATH, sample_audio, class_names)  # Tüm katmanlar
```

**Çıktılar:**
- `results/xai_gradcam/gradcam_*.png` - Tek dosya analizi
- `results/xai_gradcam/multilayer_gradcam_*.png` - Çok katmanlı analiz

---

### 2. LIME Analizi
```python
python xai_lime_analysis.py
```

**Özelleştirme:**
```python
analyze_with_lime(
    model_path='results/mfcc/mfcc_cnn_model.h5',
    audio_path='dataset/dog_bark/sample.wav',
    class_names=class_names
)
```

**Parametreler:**
- `num_samples`: LIME perturbation sayısı (varsayılan: 500)
- `num_features`: Gösterilecek segment sayısı (varsayılan: 10)

**Çıktılar:**
- `results/xai_lime/lime_*.png` - LIME segmentasyon analizi

---

### 3. Feature Importance Analizi
```python
python xai_feature_importance.py
```

**Özelleştirme:**
```python
comprehensive_feature_analysis(
    model_path='results/mfcc/mfcc_cnn_model.h5',
    audio_path='dataset/dog_bark/sample.wav',
    class_names=class_names
)
```

**Çıktılar:**
- `results/xai_feature_importance/feature_importance_*.png` - Kapsamlı analiz

---

## 📈 Görselleştirmeler

### Grad-CAM Çıktısı İçeriği:
1. ✅ Audio Waveform
2. ✅ Original MFCC
3. ✅ Grad-CAM Heatmap
4. ✅ MFCC + Heatmap Overlay
5. ✅ Class Predictions (bar chart)
6. ✅ Heatmap Statistics
7. ✅ Temporal Activation (zaman boyunca)
8. ✅ Frequency Activation (frekans boyunca)

### LIME Çıktısı İçeriği:
1. ✅ Audio Waveform
2. ✅ Original MFCC
3. ✅ LIME Explanations (top 3 sınıf için)
4. ✅ Positive Features (destekleyen özellikler)
5. ✅ Negative Features (engelleyen özellikler)
6. ✅ Feature Importance Summary

### Feature Importance Çıktısı:
1. ✅ Audio Waveform
2. ✅ Original MFCC
3. ✅ Occlusion Sensitivity Map
4. ✅ Gradient-based Importance
5. ✅ Integrated Gradients
6. ✅ Overlay Visualizations
7. ✅ Ensemble Average
8. ✅ Temporal Importance (zaman)
9. ✅ MFCC Coefficient Importance (frekans)

---

## 🔬 Matematiksel Formüller

### Grad-CAM
```
1. Forward Pass: Feature maps A^k ve predictions P al
2. Loss: L = P^c (ilgili sınıf)
3. Gradients: ∂P^c / ∂A^k
4. Weights: α_k = GAP(∂P^c / ∂A^k)
5. Weighted Sum: L^c_Grad-CAM = Σ_k α_k × A^k
6. ReLU: negatif değerleri sıfırla
7. Normalize: [0, 1] aralığına
```

### LIME
```
1. Perturb: x'₁, x'₂, ..., x'ₙ örnekleri oluştur
2. Predict: f(x'ᵢ) tahminlerini al
3. Weight: w(x'ᵢ) = exp(-d(x, x'ᵢ)²/σ²)
4. Fit: g = argmin Σᵢ w(x'ᵢ)[f(x'ᵢ) - g(x'ᵢ)]²
5. Explain: g(x) basit modelini kullan
```

### Occlusion Sensitivity
```
1. Original prediction: P_orig = f(x)
2. For each region r:
   a. Occlude: x_r ← 0
   b. Predict: P_occluded = f(x_r)
   c. Importance: I(r) = P_orig - P_occluded
3. Normalize importance map
```

### Gradient-based Importance
```
Importance = |∂P(c)/∂x|

Yüksek gradient = Input'taki küçük değişiklikler 
                   tahmini çok etkiler
```

### Integrated Gradients
```
IG_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα

Riemann sum ile:
IG ≈ (x - x') × Σ_{k=1}^m ∂F(x' + k/m(x-x'))/∂x × 1/m

x': Baseline (genellikle sıfır)
α: Interpolasyon parametresi [0,1]
m: Step sayısı
```

---

## 💡 Yorumlama Rehberi

### Grad-CAM Heatmap
- **Kırmızı/Sarı bölgeler:** Model bu bölgelere çok dikkat ediyor
- **Mavi/Siyah bölgeler:** Bu bölgeler tahmin için önemsiz
- **Temporal activation yüksek:** O zaman diliminde önemli özellikler var
- **Frequency activation yüksek:** O MFCC katsayısı önemli

### LIME Segmentleri
- **Yeşil çerçeveli:** Pozitif katkı (tahmine destek veriyor)
- **Kırmızı çerçeveli:** Negatif katkı (tahmini engelliyor)
- **Segment sayısı:** Daha fazla segment = daha detaylı analiz

### Feature Importance
- **Occlusion:** Hangi bölgelerin silinmesi tahmini en çok etkiler
- **Gradient:** Hangi bölgelerde küçük değişiklikler tahmini etkiler
- **Integrated Gradients:** En güvenilir attribution yöntemi
- **Ensemble:** Üç yöntemin ortalaması (en tutarlı sonuç)

---

## 🎯 Batch Analizi

Tüm dataset üzerinde analiz yapmak için:

```python
# xai_gradcam_analysis.py içinde:
batch_analysis(
    model_path='results/mfcc/mfcc_cnn_model.h5',
    dataset_path='dataset',
    class_names=class_names,
    samples_per_class=2  # Her sınıftan kaç örnek
)
```

Bu tüm sınıflardan örnekler alıp analiz eder.

---

## 📊 Örnek Senaryo

### Problem: Model neden "dog_bark"ı yanlış sınıflandırdı?

**Adım 1: Grad-CAM ile analiz**
```python
analyze_single_prediction(MODEL_PATH, 'dataset/dog_bark/error_sample.wav', class_names)
```
→ Model ses dalgasının ortasına değil başına odaklanmış!

**Adım 2: LIME ile detay**
```python
analyze_with_lime(MODEL_PATH, 'dataset/dog_bark/error_sample.wav', class_names)
```
→ Gürültülü bölgeler "street_music" olarak yorumlanmış!

**Adım 3: Feature Importance**
```python
comprehensive_feature_analysis(MODEL_PATH, 'dataset/dog_bark/error_sample.wav', class_names)
```
→ Düşük frekanslı MFCC katsayıları göz ardı edilmiş!

**Çözüm:** 
- Data augmentation ekle (noise)
- Düşük frekanslara daha fazla attention
- Daha fazla dog_bark örneği

---

## 🔧 Troubleshooting

### Hata: "Layer bulunamadı"
```python
# Model katmanlarını listele:
model = load_model('model.h5')
for layer in model.layers:
    print(layer.name, layer.__class__.__name__)

# Son conv layer'ı kullan:
conv_layers = [l.name for l in model.layers if 'conv' in l.name.lower()]
last_conv = conv_layers[-1]
```

### Hata: "Memory Error" (LIME)
```python
# num_samples azalt:
explanation = lime_analyzer.explain_instance(
    mfcc,
    num_samples=200,  # 1000 yerine 200
    num_features=5    # 10 yerine 5
)
```

### Hata: "Occlusion çok yavaş"
```python
# Window size büyüt:
occlusion_map = occlusion_sensitivity(
    model, mfcc, predicted_class, 
    window_size=(10, 20)  # (5, 10) yerine
)
```

---

## 📚 Referanslar

### Papers
1. **Grad-CAM:** "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (Selvaraju et al., 2017)
2. **LIME:** "Why Should I Trust You? Explaining the Predictions of Any Classifier" (Ribeiro et al., 2016)
3. **Integrated Gradients:** "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)

### Documentation
- TensorFlow: https://www.tensorflow.org/
- LIME: https://github.com/marcotcr/lime
- OpenCV: https://opencv.org/

---

## 📝 Notlar

1. **Model Bağımsız:** LIME tüm modellerde çalışır
2. **Model Bağımlı:** Grad-CAM ve Feature Importance CNN'lerde daha iyi
3. **Hız:** Grad-CAM (hızlı) > Feature Importance > LIME (yavaş)
4. **Güvenilirlik:** Integrated Gradients > Grad-CAM > Occlusion > Gradient
5. **Yorumlanabilirlik:** LIME (en kolay) > Grad-CAM > Feature Importance

---

## 🎨 Özelleştirme

### Renk Şemaları Değiştirme
```python
# Grad-CAM için
overlay = grad_cam.overlay_heatmap(heatmap, image, colormap=cv2.COLORMAP_HOT)  # JET yerine HOT

# Feature Importance için
ax.imshow(importance_map, cmap='plasma')  # 'hot' yerine 'plasma'
```

### Grafik Boyutu
```python
fig = plt.figure(figsize=(24, 16))  # Daha büyük
```

### DPI Ayarı
```python
plt.savefig(output_path, dpi=450)  # 300 yerine 450
```

---

## 🚀 Gelecek İyileştirmeler

- [ ] SHAP (SHapley Additive exPlanations) ekleme
- [ ] Attention mechanism visualization
- [ ] Interactive dashboard (Streamlit/Gradio)
- [ ] Video export (animated explanations)
- [ ] Batch comparison tools
- [ ] Adversarial examples detection

---

## 📧 İletişim

Sorularınız için GitHub Issues kullanabilirsiniz.

**Created by:** Sound Classification XAI Team
**Date:** October 2025
**Version:** 1.0

---

## ⚖️ License

MIT License - Detaylar için LICENSE dosyasına bakın.

================================================================================
