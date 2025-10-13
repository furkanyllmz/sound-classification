"""
XAI Hızlı Test - MFCC-CNN Modeli ile
Bu script, MFCC-CNN modelini kullanarak hızlı bir XAI testi yapar
"""

import os
import json
from xai_explainer import AudioXAI

def main():
    print("\n" + "="*70)
    print(" 🎯 XAI Hızlı Test - MFCC-CNN Modeli")
    print("="*70 + "\n")

    # Model kontrolü
    model_path = 'results/mfcc/mfcc_cnn_model.h5'
    results_path = 'results/mfcc/results.json'

    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        print("\n💡 İpucu: Önce modeli eğitin:")
        print("   python train_mfcc_cnn.py")
        return

    # Sınıf isimlerini yükle
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
            test_accuracy = results['test_accuracy']
        print(f"✓ Model yüklendi")
        print(f"✓ Test Accuracy: {test_accuracy:.2%}")
        print(f"✓ Sınıflar: {class_names}\n")
    else:
        print("⚠️  results.json bulunamadı, varsayılan sınıflar kullanılıyor")
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    # XAI objesi oluştur
    print("XAI modülü hazırlanıyor...")
    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    # Test ses dosyasını bul
    print("\nTest ses dosyası aranıyor...")
    audio_path = None

    # İlk sınıftan bir örnek al
    for class_name in class_names:
        class_dir = f'dataset/{class_name}'
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            if files:
                audio_path = os.path.join(class_dir, files[0])
                break

    if not audio_path or not os.path.exists(audio_path):
        print("❌ Test ses dosyası bulunamadı!")
        print("\n💡 İpucu: Ses dosyalarınızın 'dataset/' klasöründe olduğundan emin olun")
        return

    print(f"✓ Test dosyası: {audio_path}\n")

    # Kullanıcıya seçenek sun
    print("Hangi XAI analizini çalıştırmak istersiniz?")
    print("\n1. Hızlı Test (Sadece tahmin)")
    print("2. LIME Analizi")
    print("3. Zamansal Katkı Analizi")
    print("4. Frekans Katkı Analizi")
    print("5. Kapsamlı Analiz (HEPSİ)")
    print("0. Çıkış")

    choice = input("\nSeçiminiz (0-5): ").strip()

    print("\n" + "="*70 + "\n")

    if choice == '1':
        # Sadece tahmin
        result = xai.predict(audio_path)
        print(f"📁 Dosya: {os.path.basename(audio_path)}")
        print(f"🎯 Tahmin: {result['predicted_class']}")
        print(f"📊 Güven: {result['confidence']:.2%}")
        print(f"\nTüm sınıf olasılıkları:")
        for cls, prob in zip(class_names, result['predictions']):
            bar = '█' * int(prob * 50)
            print(f"  {cls:20s} {prob:6.2%} {bar}")

    elif choice == '2':
        # LIME
        print("LIME analizi çalıştırılıyor...")
        xai.explain_with_lime(audio_path, num_features=15)
        print("\n✓ LIME analizi tamamlandı!")
        print("📂 Sonuç: results/xai/lime_*.png")

    elif choice == '3':
        # Temporal
        print("Zamansal katkı analizi çalıştırılıyor...")
        contributions = xai.explain_temporal_contribution(audio_path, segment_duration=0.5)
        print("\n✓ Zamansal analiz tamamlandı!")
        print("📂 Sonuç: results/xai/temporal_*.png")
        print(f"\nSegment Katkı Skorları:")
        for i, contrib in enumerate(contributions):
            bar = '█' * int(contrib * 50)
            print(f"  Segment {i+1}: {contrib:.2%} {bar}")

    elif choice == '4':
        # Frequency
        print("Frekans katkı analizi çalıştırılıyor...")
        band_contributions = xai.explain_frequency_contribution(audio_path)
        print("\n✓ Frekans analizi tamamlandı!")
        print("📂 Sonuç: results/xai/frequency_*.png")
        print(f"\nFrekans Bandı Katkı Skorları:")
        for band_name, contrib in band_contributions:
            bar = '█' * int(contrib * 50)
            print(f"  {band_name:20s}: {contrib:.2%} {bar}")

    elif choice == '5':
        # Kapsamlı
        print("Kapsamlı XAI analizi çalıştırılıyor...")
        print("Bu işlem birkaç dakika sürebilir...\n")
        xai.comprehensive_explanation(audio_path)
        print("\n✓ Tüm analizler tamamlandı!")
        print("📂 Sonuçlar: results/xai/")

    elif choice == '0':
        print("👋 Çıkış yapılıyor...")

    else:
        print("❌ Geçersiz seçim!")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
