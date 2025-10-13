"""
XAI - Feature Importance Analizi
Hangi MFCC katsayılarının ve zaman dilimlerinin önemli olduğunu gösterir
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler

os.makedirs('results/xai_feature_importance', exist_ok=True)


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


def occlusion_sensitivity(model, mfcc, predicted_class, window_size=(5, 10)):
    """
    Occlusion Sensitivity Analizi
    
    Prensip:
    1. MFCC'nin farklı bölgelerini sıfırla (occlude)
    2. Her occluded version için tahmin yap
    3. Tahmin skorundaki düşüşü ölç
    4. En çok düşüş olan bölgeler = En önemli bölgeler
    
    Formül:
    Importance(region) = P_original(c) - P_occluded(c)
    
    P_original(c): Orijinal tahmin skoru
    P_occluded(c): Occluded tahmin skoru
    """
    h, w = mfcc.shape
    wh, ww = window_size
    
    # Orijinal tahmin
    mfcc_input = mfcc.reshape(1, h, w, 1)
    original_pred = model.predict(mfcc_input, verbose=0)[0][predicted_class]
    
    # Importance map
    importance_map = np.zeros_like(mfcc)
    
    print("   Occlusion analizi yapılıyor...")
    # Her pencereyi kaydır
    for i in range(0, h - wh + 1, wh // 2):
        for j in range(0, w - ww + 1, ww // 2):
            # MFCC kopyala
            occluded_mfcc = mfcc.copy()
            
            # Bölgeyi sıfırla
            occluded_mfcc[i:i+wh, j:j+ww] = 0
            
            # Tahmin yap
            occluded_input = occluded_mfcc.reshape(1, h, w, 1)
            occluded_pred = model.predict(occluded_input, verbose=0)[0][predicted_class]
            
            # Importance hesapla: tahmin ne kadar düştü?
            drop = original_pred - occluded_pred
            importance_map[i:i+wh, j:j+ww] += drop
    
    # Normalize
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min() + 1e-8)
    
    return importance_map


def gradient_based_importance(model, mfcc, predicted_class):
    """
    Gradient-based Feature Importance
    
    Formül:
    Importance = |∂P(c)/∂x|
    
    P(c): Sınıf c için tahmin skoru
    x: Input features (MFCC)
    
    Yüksek gradient = Input'taki küçük değişiklikler tahmini çok etkiler
    """
    import tensorflow as tf
    
    h, w = mfcc.shape
    mfcc_input = mfcc.reshape(1, h, w, 1)
    mfcc_tensor = tf.Variable(mfcc_input, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        predictions = model(mfcc_tensor)
        target_class_score = predictions[0][predicted_class]
    
    # Gradient hesapla
    gradients = tape.gradient(target_class_score, mfcc_tensor)
    gradients = gradients.numpy()[0, :, :, 0]
    
    # Absolute gradient (importance)
    importance = np.abs(gradients)
    
    # Normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance


def integrated_gradients(model, mfcc, predicted_class, steps=50):
    """
    Integrated Gradients
    
    Formül:
    IG_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i dα
    
    x: Input (MFCC)
    x': Baseline (genellikle sıfır)
    F: Model
    α: [0, 1] arasında interpolasyon
    
    Riemann toplamı ile yaklaşık hesapla:
    IG ≈ (x - x') × Σ_{k=1}^m ∂F(x' + k/m(x-x'))/∂x × 1/m
    """
    import tensorflow as tf
    
    h, w = mfcc.shape
    
    # Baseline: sıfır matrix
    baseline = np.zeros_like(mfcc)
    
    # Interpolation path oluştur
    alphas = np.linspace(0, 1, steps)
    
    integrated_grads = np.zeros_like(mfcc)
    
    print(f"   Integrated Gradients hesaplanıyor ({steps} steps)...")
    
    for alpha in alphas:
        # Interpolate
        interpolated = baseline + alpha * (mfcc - baseline)
        interpolated_input = interpolated.reshape(1, h, w, 1)
        interpolated_tensor = tf.Variable(interpolated_input, dtype=tf.float32)
        
        # Gradient hesapla
        with tf.GradientTape() as tape:
            predictions = model(interpolated_tensor)
            target_score = predictions[0][predicted_class]
        
        grads = tape.gradient(target_score, interpolated_tensor)
        grads = grads.numpy()[0, :, :, 0]
        
        # Accumulate
        integrated_grads += grads
    
    # Average and scale
    integrated_grads = integrated_grads / steps
    integrated_grads = (mfcc - baseline) * integrated_grads
    
    # Absolute value
    integrated_grads = np.abs(integrated_grads)
    
    # Normalize
    integrated_grads = (integrated_grads - integrated_grads.min()) / (integrated_grads.max() - integrated_grads.min() + 1e-8)
    
    return integrated_grads


def comprehensive_feature_analysis(model_path, audio_path, class_names):
    """
    Kapsamlı feature importance analizi
    """
    print(f"\n{'='*70}")
    print(f"Feature Importance Analizi: {os.path.basename(audio_path)}")
    print(f"{'='*70}")
    
    # Model yükle
    model = load_model(model_path)
    print(f"✓ Model yüklendi")
    
    # MFCC çıkar
    mfcc, audio, sr = extract_mfcc(audio_path)
    if mfcc is None:
        return
    
    # Tahmin yap
    mfcc_input = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    predictions = model.predict(mfcc_input, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    print(f"\n📊 Tahmin: {class_names[predicted_class]} ({confidence:.2%})")
    
    # 1. Occlusion Sensitivity
    print("\n🔍 Analiz Yöntemleri:")
    print("   1. Occlusion Sensitivity")
    occlusion_map = occlusion_sensitivity(model, mfcc, predicted_class)
    
    # 2. Gradient-based
    print("   2. Gradient-based Importance")
    gradient_map = gradient_based_importance(model, mfcc, predicted_class)
    
    # 3. Integrated Gradients
    print("   3. Integrated Gradients")
    ig_map = integrated_gradients(model, mfcc, predicted_class, steps=30)
    
    # Görselleştirme
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.3)
    
    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    librosa.display.waveshow(audio, sr=sr, ax=ax1)
    ax1.set_title('🎵 Audio Waveform', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # 2. Original MFCC
    ax2 = fig.add_subplot(gs[1, 0])
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax2, cmap='viridis')
    ax2.set_title('📈 Original MFCC', fontsize=11, fontweight='bold')
    ax2.set_ylabel('MFCC Coef.')
    plt.colorbar(ax=ax2, format='%+2.0f')
    
    # 3. Occlusion Sensitivity
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(occlusion_map, cmap='hot', aspect='auto')
    ax3.set_title('🔥 Occlusion Sensitivity', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MFCC Coef.')
    plt.colorbar(im3, ax=ax3)
    
    # 4. Gradient-based
    ax4 = fig.add_subplot(gs[1, 2])
    im4 = ax4.imshow(gradient_map, cmap='hot', aspect='auto')
    ax4.set_title('📊 Gradient-based', fontsize=11, fontweight='bold')
    ax4.set_ylabel('MFCC Coef.')
    plt.colorbar(im4, ax=ax4)
    
    # 5. Integrated Gradients
    ax5 = fig.add_subplot(gs[1, 3])
    im5 = ax5.imshow(ig_map, cmap='hot', aspect='auto')
    ax5.set_title('⚡ Integrated Gradients', fontsize=11, fontweight='bold')
    ax5.set_ylabel('MFCC Coef.')
    plt.colorbar(im5, ax=ax5)
    
    # 6-8. Overlay visualizations
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.imshow(mfcc, cmap='gray', aspect='auto', alpha=0.5)
    ax6.imshow(occlusion_map, cmap='hot', aspect='auto', alpha=0.5)
    ax6.set_title('Occlusion Overlay', fontsize=10, fontweight='bold')
    ax6.set_ylabel('MFCC Coef.')
    
    ax7 = fig.add_subplot(gs[2, 1])
    ax7.imshow(mfcc, cmap='gray', aspect='auto', alpha=0.5)
    ax7.imshow(gradient_map, cmap='hot', aspect='auto', alpha=0.5)
    ax7.set_title('Gradient Overlay', fontsize=10, fontweight='bold')
    ax7.set_ylabel('MFCC Coef.')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.imshow(mfcc, cmap='gray', aspect='auto', alpha=0.5)
    ax8.imshow(ig_map, cmap='hot', aspect='auto', alpha=0.5)
    ax8.set_title('IG Overlay', fontsize=10, fontweight='bold')
    ax8.set_ylabel('MFCC Coef.')
    
    # 9. Ensemble (average)
    ensemble_map = (occlusion_map + gradient_map + ig_map) / 3
    ax9 = fig.add_subplot(gs[2, 3])
    im9 = ax9.imshow(ensemble_map, cmap='hot', aspect='auto')
    ax9.set_title('🎯 Ensemble Average', fontsize=10, fontweight='bold')
    ax9.set_ylabel('MFCC Coef.')
    plt.colorbar(im9, ax=ax9)
    
    # 10-11. Temporal importance
    ax10 = fig.add_subplot(gs[3, :2])
    temporal_occlusion = np.mean(occlusion_map, axis=0)
    temporal_gradient = np.mean(gradient_map, axis=0)
    temporal_ig = np.mean(ig_map, axis=0)
    
    ax10.plot(temporal_occlusion, label='Occlusion', linewidth=2, alpha=0.7)
    ax10.plot(temporal_gradient, label='Gradient', linewidth=2, alpha=0.7)
    ax10.plot(temporal_ig, label='Int. Gradients', linewidth=2, alpha=0.7)
    ax10.fill_between(range(len(temporal_occlusion)), temporal_occlusion, alpha=0.2)
    ax10.set_title('⏱️ Temporal Importance (averaged over MFCC coefficients)', fontsize=11, fontweight='bold')
    ax10.set_xlabel('Time Frames')
    ax10.set_ylabel('Mean Importance')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 12. Frequency importance
    ax11 = fig.add_subplot(gs[3, 2:])
    freq_occlusion = np.mean(occlusion_map, axis=1)
    freq_gradient = np.mean(gradient_map, axis=1)
    freq_ig = np.mean(ig_map, axis=1)
    
    y_pos = np.arange(len(freq_occlusion))
    width = 0.25
    
    ax11.barh(y_pos - width, freq_occlusion, width, label='Occlusion', alpha=0.7)
    ax11.barh(y_pos, freq_gradient, width, label='Gradient', alpha=0.7)
    ax11.barh(y_pos + width, freq_ig, width, label='Int. Gradients', alpha=0.7)
    ax11.set_title('🎼 MFCC Coefficient Importance', fontsize=11, fontweight='bold')
    ax11.set_xlabel('Mean Importance')
    ax11.set_ylabel('MFCC Coefficient Index')
    ax11.legend()
    ax11.grid(axis='x', alpha=0.3)
    ax11.invert_yaxis()
    
    # 13. Predictions
    ax12 = fig.add_subplot(gs[4, :])
    bars = ax12.barh(class_names, predictions[0], color='steelblue', edgecolor='black')
    bars[predicted_class].set_color('crimson')
    ax12.set_xlabel('Confidence Score', fontsize=12)
    ax12.set_title('📊 Class Predictions', fontsize=12, fontweight='bold')
    ax12.set_xlim([0, 1])
    ax12.grid(axis='x', alpha=0.3)
    
    for i, (name, score) in enumerate(zip(class_names, predictions[0])):
        ax12.text(score + 0.02, i, f'{score:.2%}', va='center', fontweight='bold')
    
    # Ana başlık
    fig.suptitle(f'Feature Importance Analysis: {class_names[predicted_class]} ({confidence:.2%})',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Kaydet
    output_path = f'results/xai_feature_importance/feature_importance_{os.path.basename(audio_path).replace(".wav", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Feature importance analizi kaydedildi: {output_path}")
    plt.close()


if __name__ == "__main__":
    class_names = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    
    MODEL_PATH = 'results/mfcc/mfcc_cnn_model.h5'
    sample_audio = 'dataset/dog_bark/dog_bark_1.wav'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(sample_audio):
        comprehensive_feature_analysis(MODEL_PATH, sample_audio, class_names)
    else:
        print("⚠️ Model veya ses dosyası bulunamadı!")
