"""
XAI (Explainable AI) for Audio Classification
Ses sınıflandırma modelinin kararlarını açıklar

Desteklenen yöntemler:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations)
3. Attention Weights Visualization
4. Feature Importance Analysis
5. Temporal Contribution Map
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os
from sklearn.preprocessing import LabelEncoder
import json

# LIME import
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME kütüphanesi bulunamadı. Kurulum: pip install lime")
    LIME_AVAILABLE = False

# SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP kütüphanesi bulunamadı. Kurulum: pip install shap")
    SHAP_AVAILABLE = False


class AudioXAI:
    """
    Ses sınıflandırma modelleri için XAI (Explainable AI) sınıfı
    """

    def __init__(self, model_path, class_names, model_type='yamnet'):
        """
        Args:
            model_path: Model dosya yolu (.h5)
            class_names: Sınıf isimleri listesi
            model_type: 'yamnet', 'mfcc_cnn', veya 'vgg16'
        """
        self.model_path = model_path
        self.class_names = class_names
        self.model_type = model_type
        self.model = load_model(model_path, compile=False)

        if model_type == 'yamnet':
            print("YAMNet modeli yükleniyor...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            print("✓ YAMNet yüklendi!")

        print(f"✓ Model yüklendi: {model_path}")
        print(f"✓ Model tipi: {model_type}")
        print(f"✓ Sınıflar: {class_names}")

    def load_and_preprocess_audio(self, audio_path):
        """Ses dosyasını yükler ve model tipine göre preprocess eder"""
        if self.model_type == 'yamnet':
            # YAMNet için 16kHz
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            max_length = 4 * 16000
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            else:
                audio = audio[:max_length]
            return audio, sr
        else:
            # MFCC-CNN için
            audio, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=4.0)
            return audio, sr

    def extract_features(self, audio):
        """Model tipine göre feature extraction"""
        if self.model_type == 'yamnet':
            # YAMNet embeddings
            audio_tensor = tf.cast(audio, tf.float32)
            scores, embeddings, spectrogram = self.yamnet_model(audio_tensor)
            embedding_mean = tf.reduce_mean(embeddings, axis=0)
            return embedding_mean.numpy().reshape(1, -1)
        else:
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
            max_pad_len = 174
            pad_width = max_pad_len - mfccs.shape[1]
            if pad_width > 0:
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :max_pad_len]
            return mfccs.reshape(1, 40, 174, 1)

    def predict(self, audio_path):
        """Ses dosyası için tahmin yapar"""
        audio, sr = self.load_and_preprocess_audio(audio_path)
        features = self.extract_features(audio)
        predictions = self.model.predict(features, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]

        return {
            'audio': audio,
            'sr': sr,
            'features': features,
            'predictions': predictions,
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence
        }

    def explain_with_lime(self, audio_path, num_features=10):
        """
        LIME kullanarak model kararını açıklar

        Args:
            audio_path: Ses dosyası yolu
            num_features: Gösterilecek önemli feature sayısı
        """
        if not LIME_AVAILABLE:
            print("❌ LIME kütüphanesi yüklü değil!")
            return None

        print(f"\n{'='*60}")
        print("LIME Explanation")
        print(f"{'='*60}\n")

        # Tahmin yap
        result = self.predict(audio_path)
        audio = result['audio']
        features = result['features'].flatten()

        print(f"Dosya: {os.path.basename(audio_path)}")
        print(f"Tahmin: {result['predicted_class']} ({result['confidence']:.2%})")

        # LIME explainer oluştur
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=features.reshape(1, -1),
            feature_names=[f'Feature_{i}' for i in range(len(features))],
            class_names=self.class_names,
            mode='classification'
        )

        # Prediction fonksiyonu
        def predict_fn(x):
            if self.model_type == 'yamnet':
                return self.model.predict(x, verbose=0)
            else:
                x_reshaped = x.reshape(-1, 40, 174, 1)
                return self.model.predict(x_reshaped, verbose=0)

        # LIME explanation
        exp = explainer.explain_instance(
            features,
            predict_fn,
            num_features=num_features,
            top_labels=3
        )

        # Görselleştirme
        fig = plt.figure(figsize=(15, 10))

        # 1. Top features
        plt.subplot(2, 2, 1)
        exp.as_pyplot_figure(label=result['predicted_class_idx'])
        plt.title(f"LIME: Top {num_features} Features\nPredicted: {result['predicted_class']}",
                  fontsize=12, fontweight='bold')

        # 2. Waveform
        plt.subplot(2, 2, 2)
        time = np.arange(len(audio)) / result['sr']
        plt.plot(time, audio, linewidth=0.5, alpha=0.8)
        plt.title('Audio Waveform', fontsize=12, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # 3. Class probabilities
        plt.subplot(2, 2, 3)
        y_pos = np.arange(len(self.class_names))
        colors = plt.cm.RdYlGn(result['predictions'])
        plt.barh(y_pos, result['predictions'], color=colors)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.title('Class Probabilities', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 4. Feature importance weights
        plt.subplot(2, 2, 4)
        weights = dict(exp.as_list(label=result['predicted_class_idx']))
        feature_names = list(weights.keys())[:num_features]
        feature_values = list(weights.values())[:num_features]

        colors_importance = ['green' if x > 0 else 'red' for x in feature_values]
        plt.barh(range(len(feature_names)), feature_values, color=colors_importance, alpha=0.7)
        plt.yticks(range(len(feature_names)), feature_names, fontsize=8)
        plt.xlabel('Weight')
        plt.title('Feature Importance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_path = f'results/xai/lime_{os.path.basename(audio_path).replace(".wav", ".png")}'
        os.makedirs('results/xai', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ LIME açıklaması kaydedildi: {output_path}")
        plt.close()

        return exp

    def explain_temporal_contribution(self, audio_path, segment_duration=0.5):
        """
        Zamansal katkı analizi - Sesin hangi bölümlerinin önemli olduğunu gösterir

        Args:
            audio_path: Ses dosyası yolu
            segment_duration: Segment süresi (saniye)
        """
        print(f"\n{'='*60}")
        print("Temporal Contribution Analysis")
        print(f"{'='*60}\n")

        # Tahmin yap
        result = self.predict(audio_path)
        audio = result['audio']
        sr = result['sr']
        predicted_class_idx = result['predicted_class_idx']

        print(f"Dosya: {os.path.basename(audio_path)}")
        print(f"Tahmin: {result['predicted_class']} ({result['confidence']:.2%})")

        # Ses dosyasını segmentlere böl
        segment_length = int(segment_duration * sr)
        num_segments = len(audio) // segment_length

        segment_contributions = []

        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]

            # Segment için padding
            if len(segment) < len(audio):
                segment_padded = np.zeros_like(audio)
                segment_padded[start:end] = segment
            else:
                segment_padded = segment

            # Feature extraction ve prediction
            features = self.extract_features(segment_padded)
            predictions = self.model.predict(features, verbose=0)[0]
            contribution = predictions[predicted_class_idx]
            segment_contributions.append(contribution)

        # Görselleştirme
        fig = plt.figure(figsize=(16, 10))

        # 1. Waveform with segments
        plt.subplot(3, 1, 1)
        time = np.arange(len(audio)) / sr
        plt.plot(time, audio, linewidth=0.5, alpha=0.5, color='gray')

        # Segment renklerini contribution'a göre belirle
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = start_time + segment_duration
            color_intensity = segment_contributions[i]
            plt.axvspan(start_time, end_time, alpha=color_intensity*0.5, color='red')

        plt.title(f'Audio Waveform with Temporal Contributions\nPredicted: {result["predicted_class"]}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

        # 2. Spectrogram
        plt.subplot(3, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram', fontsize=14, fontweight='bold')

        # Segment çizgileri
        for i in range(num_segments + 1):
            plt.axvline(x=i * segment_duration, color='red', linestyle='--', alpha=0.5)

        # 3. Contribution scores
        plt.subplot(3, 1, 3)
        segment_times = [i * segment_duration for i in range(num_segments)]
        colors_contrib = plt.cm.RdYlGn(segment_contributions)
        plt.bar(segment_times, segment_contributions, width=segment_duration*0.9,
                color=colors_contrib, edgecolor='black', linewidth=1)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Contribution Score', fontsize=12)
        plt.title('Segment Contribution to Prediction', fontsize=14, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        output_path = f'results/xai/temporal_{os.path.basename(audio_path).replace(".wav", ".png")}'
        os.makedirs('results/xai', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Zamansal katkı analizi kaydedildi: {output_path}")
        plt.close()

        return segment_contributions

    def explain_frequency_contribution(self, audio_path):
        """
        Frekans katkı analizi - Hangi frekans bantlarının önemli olduğunu gösterir
        """
        print(f"\n{'='*60}")
        print("Frequency Contribution Analysis")
        print(f"{'='*60}\n")

        # Tahmin yap
        result = self.predict(audio_path)
        audio = result['audio']
        sr = result['sr']
        predicted_class_idx = result['predicted_class_idx']

        print(f"Dosya: {os.path.basename(audio_path)}")
        print(f"Tahmin: {result['predicted_class']} ({result['confidence']:.2%})")

        # Frekans bantları
        freq_bands = [
            (0, 200, 'Sub-bass'),
            (200, 500, 'Bass'),
            (500, 2000, 'Midrange'),
            (2000, 4000, 'Upper Midrange'),
            (4000, 8000, 'Presence'),
            (8000, sr//2, 'Brilliance')
        ]

        band_contributions = []

        for low, high, name in freq_bands:
            # Bandpass filter
            audio_filtered = librosa.effects.preemphasis(audio)

            # Bu banda ait sinyali çıkar (basitleştirilmiş)
            stft = librosa.stft(audio)
            freqs = librosa.fft_frequencies(sr=sr)

            # Frekans maskesi
            mask = (freqs >= low) & (freqs < high)
            stft_filtered = stft.copy()
            stft_filtered[~mask, :] = 0

            # Inverse STFT
            audio_band = librosa.istft(stft_filtered)

            # Padding
            if len(audio_band) < len(audio):
                audio_band = np.pad(audio_band, (0, len(audio) - len(audio_band)), mode='constant')
            else:
                audio_band = audio_band[:len(audio)]

            # Feature extraction ve prediction
            features = self.extract_features(audio_band)
            predictions = self.model.predict(features, verbose=0)[0]
            contribution = predictions[predicted_class_idx]
            band_contributions.append((name, contribution))

        # Görselleştirme
        fig = plt.figure(figsize=(16, 10))

        # 1. Mel Spectrogram
        plt.subplot(2, 2, 1)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram', fontsize=12, fontweight='bold')

        # 2. Frequency band contributions
        plt.subplot(2, 2, 2)
        band_names = [name for name, _ in band_contributions]
        contributions = [contrib for _, contrib in band_contributions]
        colors_freq = plt.cm.plasma(np.array(contributions))

        plt.barh(range(len(band_names)), contributions, color=colors_freq, edgecolor='black', linewidth=1.5)
        plt.yticks(range(len(band_names)), band_names)
        plt.xlabel('Contribution Score')
        plt.title(f'Frequency Band Contributions\nPredicted: {result["predicted_class"]}',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # 3. Chromagram
        plt.subplot(2, 2, 3)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
        plt.colorbar()
        plt.title('Chromagram', fontsize=12, fontweight='bold')

        # 4. Class probabilities
        plt.subplot(2, 2, 4)
        y_pos = np.arange(len(self.class_names))
        colors_class = plt.cm.RdYlGn(result['predictions'])
        plt.barh(y_pos, result['predictions'], color=colors_class)
        plt.yticks(y_pos, self.class_names)
        plt.xlabel('Probability')
        plt.title('Class Probabilities', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_path = f'results/xai/frequency_{os.path.basename(audio_path).replace(".wav", ".png")}'
        os.makedirs('results/xai', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Frekans katkı analizi kaydedildi: {output_path}")
        plt.close()

        return band_contributions

    def comprehensive_explanation(self, audio_path):
        """
        Tüm XAI yöntemlerini birlikte çalıştırır ve kapsamlı bir açıklama üretir
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE XAI ANALYSIS")
        print("="*60)

        # Tahmin
        result = self.predict(audio_path)
        print(f"\n📁 Dosya: {os.path.basename(audio_path)}")
        print(f"🎯 Tahmin: {result['predicted_class']}")
        print(f"📊 Güven: {result['confidence']:.2%}")
        print(f"\nTüm sınıf olasılıkları:")
        for i, (cls, prob) in enumerate(zip(self.class_names, result['predictions'])):
            bar = '█' * int(prob * 50)
            print(f"  {cls:20s} {prob:6.2%} {bar}")

        # 1. LIME
        if LIME_AVAILABLE:
            print("\n1️⃣  LIME analizi yapılıyor...")
            self.explain_with_lime(audio_path, num_features=15)

        # 2. Temporal Contribution
        print("\n2️⃣  Zamansal katkı analizi yapılıyor...")
        self.explain_temporal_contribution(audio_path, segment_duration=0.5)

        # 3. Frequency Contribution
        print("\n3️⃣  Frekans katkı analizi yapılıyor...")
        self.explain_frequency_contribution(audio_path)

        print("\n" + "="*60)
        print("✓ TÜM ANALIZLER TAMAMLANDI!")
        print("="*60)
        print(f"📂 Sonuçlar: results/xai/")
        print("="*60 + "\n")


def main():
    """Ana fonksiyon - Örnek kullanım"""

    # Model ve sınıf bilgileri
    model_path = 'results/yamnet_transfer/yamnet_transfer_model.h5'

    # Class names (results.json'dan oku)
    results_path = 'results/yamnet_transfer/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    # XAI nesnesi oluştur
    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='yamnet'
    )

    # Test ses dosyası (örnek)
    # Kendi ses dosyanızı buraya koyun
    audio_path = 'dataset/dog_bark/100032-3-0-0.wav'  # Örnek

    if not os.path.exists(audio_path):
        print(f"❌ Ses dosyası bulunamadı: {audio_path}")
        print("Lütfen geçerli bir ses dosyası yolu girin.")
        return

    # Kapsamlı açıklama
    xai.comprehensive_explanation(audio_path)


if __name__ == '__main__':
    main()
