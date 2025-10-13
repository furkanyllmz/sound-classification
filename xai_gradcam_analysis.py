"""
XAI - Grad-CAM Analizi
Ses Sınıflandırma Modelleri için Açıklanabilir AI
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder

# Sonuçlar için klasör
os.makedirs('results/xai_gradcam', exist_ok=True)

class GradCAM:
    """
    Grad-CAM implementasyonu
    
    Formül:
    L_Grad-CAM = ReLU(Σ_k α_k * A_k)
    
    α_k = (1/Z) * Σ_i Σ_j (∂y^c / ∂A_k_ij)
    
    y^c: Sınıf c için skor
    A_k: k'ıncı feature map
    α_k: Feature map k'nın ağırlığı
    """
    
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        
        # Grad-CAM modeli oluştur
        self.grad_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
    
    def compute_heatmap(self, image, class_idx, eps=1e-8):
        """
        Grad-CAM heatmap hesapla
        
        Adımlar:
        1. Forward pass: feature maps ve predictions al
        2. Loss hesapla: L = y^c (sadece ilgili sınıf)
        3. Backward pass: gradient hesapla
        4. Global Average Pooling: α_k = GAP(gradients)
        5. Weighted combination: Σ_k α_k * A_k
        6. ReLU aktivasyon
        7. Normalize [0, 1]
        """
        # Gradient tape ile forward pass
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            # İlgili sınıfın skorunu al
            loss = predictions[:, class_idx]
        
        # Gradient hesapla: ∂y^c / ∂A
        grads = tape.gradient(loss, conv_outputs)
        
        # Global Average Pooling: α_k = (1/Z) * Σ_i Σ_j (∂y^c / ∂A_k_ij)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Numpy'a çevir
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Weighted combination: Σ_k α_k * A_k
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Channel boyunca topla
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU: negatif değerleri sıfırla
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize [0, 1]
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Heatmap'i orijinal görüntü üzerine bindir
        
        Formül:
        Output = α * Heatmap + (1-α) * Image
        """
        # Heatmap'i image boyutuna resize et
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Colormap uygula
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Image'i normalize et ve 3 kanala çevir
        image_normalized = image.squeeze()
        if len(image_normalized.shape) == 2:
            image_normalized = np.stack([image_normalized] * 3, axis=-1)
        
        image_normalized = (image_normalized - image_normalized.min()) / (image_normalized.max() - image_normalized.min() + 1e-8)
        image_normalized = np.uint8(255 * image_normalized)
        
        # Blend: α * heatmap + (1-α) * image
        overlay = cv2.addWeighted(heatmap, alpha, image_normalized, 1 - alpha, 0)
        
        return overlay


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


def analyze_single_prediction(model_path, audio_path, class_names, layer_name='conv2d_5'):
    """
    Tek bir ses dosyası için Grad-CAM analizi
    """
    print(f"\n{'='*70}")
    print(f"Grad-CAM Analizi: {os.path.basename(audio_path)}")
    print(f"{'='*70}")
    
    # Model yükle
    model = load_model(model_path)
    print(f"✓ Model yüklendi: {model_path}")
    
    # Konvolüsyon katmanlarını listele
    conv_layers = [layer.name for layer in model.layers if 'conv2d' in layer.name.lower()]
    print(f"✓ Konvolüsyon katmanları: {conv_layers}")
    
    if layer_name not in conv_layers:
        layer_name = conv_layers[-1]  # Son conv katmanını kullan
        print(f"⚠ Layer bulunamadı, son katman kullanılıyor: {layer_name}")
    
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
    
    print(f"\n📊 Tahmin Sonuçları:")
    print(f"   Tahmin edilen sınıf: {class_names[predicted_class]}")
    print(f"   Güven skoru: {confidence:.2%}")
    
    # Grad-CAM oluştur
    grad_cam = GradCAM(model, layer_name)
    heatmap = grad_cam.compute_heatmap(mfcc_input, predicted_class)
    
    # Overlay oluştur
    overlay = grad_cam.overlay_heatmap(heatmap, mfcc)
    
    # Görselleştirme
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title('🎵 Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # 2. MFCC
    ax2 = fig.add_subplot(gs[1, 0])
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax2, cmap='viridis')
    ax2.set_title('📈 Original MFCC', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MFCC Coefficients')
    plt.colorbar(ax=ax2, format='%+2.0f')
    
    # 3. Grad-CAM Heatmap
    ax3 = fig.add_subplot(gs[1, 1])
    im = ax3.imshow(heatmap, cmap='jet', aspect='auto')
    ax3.set_title('🔥 Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Frames')
    ax3.set_ylabel('Feature Map')
    plt.colorbar(im, ax=ax3)
    
    # 4. Overlay
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.imshow(overlay)
    ax4.set_title('🎯 MFCC + Grad-CAM Overlay', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time Frames')
    ax4.set_ylabel('MFCC Coefficients')
    ax4.axis('off')
    
    # 5. Prediction Confidence
    ax5 = fig.add_subplot(gs[2, :])
    bars = ax5.barh(class_names, predictions[0], color='steelblue', edgecolor='black')
    bars[predicted_class].set_color('crimson')
    ax5.set_xlabel('Confidence Score', fontsize=12)
    ax5.set_title('📊 Class Predictions', fontsize=12, fontweight='bold')
    ax5.set_xlim([0, 1])
    ax5.grid(axis='x', alpha=0.3)
    
    # Değerleri bar'ların üzerine yaz
    for i, (name, score) in enumerate(zip(class_names, predictions[0])):
        ax5.text(score + 0.02, i, f'{score:.2%}', va='center', fontweight='bold')
    
    # 6. Heatmap Statistics
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.axis('off')
    stats_text = f"""
    📈 Grad-CAM İstatistikleri:
    
    Max Activation: {heatmap.max():.4f}
    Mean Activation: {heatmap.mean():.4f}
    Std Activation: {heatmap.std():.4f}
    
    Layer: {layer_name}
    Shape: {heatmap.shape}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Temporal Activation
    ax7 = fig.add_subplot(gs[3, 1])
    temporal_activation = np.mean(heatmap, axis=0)
    ax7.plot(temporal_activation, linewidth=2, color='crimson')
    ax7.fill_between(range(len(temporal_activation)), temporal_activation, alpha=0.3, color='crimson')
    ax7.set_title('⏱️ Temporal Activation', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Time Frames')
    ax7.set_ylabel('Mean Activation')
    ax7.grid(True, alpha=0.3)
    
    # 8. Frequency Activation
    ax8 = fig.add_subplot(gs[3, 2])
    frequency_activation = np.mean(heatmap, axis=1)
    ax8.barh(range(len(frequency_activation)), frequency_activation, color='steelblue', edgecolor='black')
    ax8.set_title('🎼 Frequency Activation', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Mean Activation')
    ax8.set_ylabel('Feature Dimension')
    ax8.grid(axis='x', alpha=0.3)
    ax8.invert_yaxis()
    
    # Ana başlık
    fig.suptitle(f'Grad-CAM Analysis: {class_names[predicted_class]} ({confidence:.2%} confidence)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Kaydet
    output_path = f'results/xai_gradcam/gradcam_{os.path.basename(audio_path).replace(".wav", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Grad-CAM analizi kaydedildi: {output_path}")
    plt.close()


def analyze_multiple_layers(model_path, audio_path, class_names):
    """
    Birden fazla katman için Grad-CAM analizi
    """
    print(f"\n{'='*70}")
    print(f"Multi-Layer Grad-CAM Analizi")
    print(f"{'='*70}")
    
    # Model yükle
    model = load_model(model_path)
    
    # Tüm conv katmanlarını al
    conv_layers = [layer.name for layer in model.layers if 'conv2d' in layer.name.lower()]
    print(f"✓ Analiz edilecek katmanlar: {conv_layers}")
    
    # MFCC çıkar
    mfcc, audio, sr = extract_mfcc(audio_path)
    if mfcc is None:
        return
    
    mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    predictions = model.predict(mfcc_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    
    # Her katman için Grad-CAM
    n_layers = len(conv_layers)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, layer_name in enumerate(conv_layers):
        grad_cam = GradCAM(model, layer_name)
        heatmap = grad_cam.compute_heatmap(mfcc_input, predicted_class)
        
        axes[idx].imshow(heatmap, cmap='jet', aspect='auto')
        axes[idx].set_title(f'{layer_name}\n(shape: {heatmap.shape})', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Boş subplotları gizle
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Multi-Layer Grad-CAM: {class_names[predicted_class]}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = f'results/xai_gradcam/multilayer_gradcam_{os.path.basename(audio_path).replace(".wav", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Multi-layer analizi kaydedildi: {output_path}")
    plt.close()


def batch_analysis(model_path, dataset_path, class_names, samples_per_class=2):
    """
    Her sınıftan örnek alarak batch Grad-CAM analizi
    """
    print(f"\n{'='*70}")
    print(f"Batch Grad-CAM Analizi")
    print(f"{'='*70}")
    
    model = load_model(model_path)
    conv_layers = [layer.name for layer in model.layers if 'conv2d' in layer.name.lower()]
    last_conv = conv_layers[-1]
    
    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue
        
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')][:samples_per_class]
        
        for wav_file in wav_files:
            audio_path = os.path.join(class_path, wav_file)
            analyze_single_prediction(model_path, audio_path, class_names, last_conv)
    
    print(f"\n{'='*70}")
    print(f"✅ Batch analizi tamamlandı!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Örnek kullanım
    
    # Sınıf isimleri (kendi dataset'inize göre değiştirin)
    class_names = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    
    # Model ve veri yolları
    MODEL_PATH = 'results/mfcc/mfcc_cnn_model.h5'
    DATASET_PATH = 'dataset'
    
    # Tek dosya analizi
    sample_audio = 'dataset/dog_bark/dog_bark_1.wav'  # Kendi dosyanızı ekleyin
    
    if os.path.exists(MODEL_PATH) and os.path.exists(sample_audio):
        analyze_single_prediction(MODEL_PATH, sample_audio, class_names)
        analyze_multiple_layers(MODEL_PATH, sample_audio, class_names)
    else:
        print("⚠️ Model veya ses dosyası bulunamadı!")
        print(f"Model: {MODEL_PATH}")
        print(f"Audio: {sample_audio}")
    
    # Batch analizi (opsiyonel)
    # batch_analysis(MODEL_PATH, DATASET_PATH, class_names, samples_per_class=1)
