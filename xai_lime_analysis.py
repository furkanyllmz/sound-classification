"""
XAI - LIME (Local Interpretable Model-agnostic Explanations)
Ses Sınıflandırma için Model-Agnostik Açıklama
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from lime import lime_image
from skimage.segmentation import mark_boundaries

os.makedirs('results/xai_lime', exist_ok=True)

class AudioLIME:
    """
    LIME implementasyonu ses verileri için
    
    LIME Prensibi:
    1. Orijinal örneği perturb et (bozarak yeni örnekler oluştur)
    2. Her perturbation için model tahminini al
    3. Orijinal örneğe yakın örneklere daha fazla ağırlık ver
    4. Basit bir model (linear) ile açıkla
    
    Formül:
    ξ(x) = argmin_{g∈G} L(f, g, π_x) + Ω(g)
    
    L: Loss (model f ile açıklayıcı g arasındaki fark)
    π_x: Proximity measure (orijinale yakınlık)
    Ω(g): Model karmaşıklığı (regularization)
    """
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
    
    def predict_fn(self, images):
        """
        LIME için tahmin fonksiyonu
        
        Args:
            images: (batch_size, height, width, channels)
        
        Returns:
            predictions: (batch_size, num_classes)
        """
        # Model input shape'ine uygun hale getir
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        predictions = self.model.predict(images, verbose=0)
        return predictions
    
    def explain_instance(self, mfcc, top_labels=3, num_samples=1000, num_features=10):
        """
        LIME ile tek bir örneği açıkla
        
        Args:
            mfcc: MFCC features (height, width)
            top_labels: Kaç sınıf için açıklama üret
            num_samples: Kaç perturbation örneği oluştur
            num_features: Kaç önemli segment göster
        
        Returns:
            explanation: LIME explanation object
        """
        # LIME explainer oluştur
        explainer = lime_image.LimeImageExplainer()
        
        # MFCC'yi 3 kanala çevir (LIME RGB bekliyor)
        mfcc_rgb = np.stack([mfcc] * 3, axis=-1)
        
        # Normalize [0, 1]
        mfcc_normalized = (mfcc_rgb - mfcc_rgb.min()) / (mfcc_rgb.max() - mfcc_rgb.min() + 1e-8)
        
        # Explanation oluştur
        explanation = explainer.explain_instance(
            mfcc_normalized,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            random_seed=42
        )
        
        return explanation


def extract_mfcc(file_path, max_pad_len=174):
    """MFCC özelliklerini çıkar"""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=4.0)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs, audio, sample_rate
    except Exception as e:
        print(f"Hata: {e}")
        return None, None, None


def analyze_with_lime(model_path, audio_path, class_names):
    """
    LIME ile ses analizi
    """
    print(f"\n{'='*70}")
    print(f"LIME Analizi: {os.path.basename(audio_path)}")
    print(f"{'='*70}")
    
    # Model yükle
    model = load_model(model_path)
    print(f"✓ Model yüklendi")
    
    # MFCC çıkar
    mfcc, audio, sr = extract_mfcc(audio_path)
    if mfcc is None:
        print("❌ MFCC çıkarılamadı!")
        return
    
    # Model için hazırla
    mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    
    # Tahmin yap
    predictions = model.predict(mfcc_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"\n📊 Tahmin: {class_names[predicted_class]} ({confidence:.2%})")
    
    # LIME analizi
    print(f"\n🔍 LIME analizi yapılıyor...")
    lime_analyzer = AudioLIME(model, class_names)
    explanation = lime_analyzer.explain_instance(
        mfcc,
        top_labels=3,
        num_samples=500,
        num_features=10
    )
    
    # Görselleştirme
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title('🎵 Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # 2. Original MFCC
    ax2 = fig.add_subplot(gs[1, 0])
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax2, cmap='viridis')
    ax2.set_title('📈 Original MFCC', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MFCC Coefficients')
    plt.colorbar(ax=ax2, format='%+2.0f')
    
    # 3-5. LIME Explanations for top 3 classes
    top_labels = explanation.top_labels[:3]
    
    for idx, label in enumerate(top_labels):
        ax = fig.add_subplot(gs[1, idx])
        
        # LIME segmentlerini al
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        # Boundaries ile işaretle
        boundaries = mark_boundaries(temp, mask)
        
        ax.imshow(boundaries)
        ax.set_title(f'🔍 LIME: {class_names[label]}\n(prob: {predictions[0][label]:.2%})',
                    fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # 6. Prediction probabilities
    ax6 = fig.add_subplot(gs[2, :])
    bars = ax6.barh(class_names, predictions[0], color='steelblue', edgecolor='black')
    bars[predicted_class].set_color('crimson')
    ax6.set_xlabel('Confidence Score', fontsize=12)
    ax6.set_title('📊 Class Predictions', fontsize=12, fontweight='bold')
    ax6.set_xlim([0, 1])
    ax6.grid(axis='x', alpha=0.3)
    
    for i, (name, score) in enumerate(zip(class_names, predictions[0])):
        ax6.text(score + 0.02, i, f'{score:.2%}', va='center', fontweight='bold')
    
    # 7-9. Positive/Negative contributions for predicted class
    ax7 = fig.add_subplot(gs[3, 0])
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=True,
        num_features=10,
        hide_rest=False,
        min_weight=0.01
    )
    boundaries_pos = mark_boundaries(temp, mask)
    ax7.imshow(boundaries_pos)
    ax7.set_title(f'✅ Positive Features\n{class_names[predicted_class]}', 
                  fontsize=11, fontweight='bold', color='green')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[3, 1])
    temp, mask = explanation.get_image_and_mask(
        predicted_class,
        positive_only=False,
        num_features=10,
        hide_rest=False,
        min_weight=0.01
    )
    boundaries_neg = mark_boundaries(temp, mask)
    ax8.imshow(boundaries_neg)
    ax8.set_title(f'❌ Negative Features\n{class_names[predicted_class]}', 
                  fontsize=11, fontweight='bold', color='red')
    ax8.axis('off')
    
    # 10. Feature importance
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')
    
    # Feature weights al
    local_exp = explanation.local_exp[predicted_class]
    local_exp_sorted = sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    info_text = f"""
    🎯 LIME Analysis Summary
    
    Predicted: {class_names[predicted_class]}
    Confidence: {confidence:.2%}
    
    Top 5 Segments (by importance):
    """
    
    for segment, weight in local_exp_sorted:
        sign = "+" if weight > 0 else "-"
        info_text += f"\n   {sign} Segment {segment}: {abs(weight):.4f}"
    
    ax9.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    # Ana başlık
    fig.suptitle(f'LIME Analysis: {class_names[predicted_class]} ({confidence:.2%} confidence)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Kaydet
    output_path = f'results/xai_lime/lime_{os.path.basename(audio_path).replace(".wav", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ LIME analizi kaydedildi: {output_path}")
    plt.close()
    
    return explanation


if __name__ == "__main__":
    # Örnek kullanım
    
    class_names = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    
    MODEL_PATH = 'results/mfcc/mfcc_cnn_model.h5'
    sample_audio = 'dataset/dog_bark/dog_bark_1.wav'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(sample_audio):
        analyze_with_lime(MODEL_PATH, sample_audio, class_names)
    else:
        print("⚠️ Model veya ses dosyası bulunamadı!")
        print(f"Model: {MODEL_PATH}")
        print(f"Audio: {sample_audio}")
