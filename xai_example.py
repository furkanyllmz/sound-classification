"""
XAI (Explainable AI) Kullanım Örneği

Bu script, eğitilmiş bir modelin kararlarını açıklamak için
farklı XAI yöntemlerini nasıl kullanacağınızı gösterir.
"""

import os
import json
from xai_explainer import AudioXAI

def example_single_explanation():
    """Tek bir ses dosyası için açıklama"""

    print("\n" + "="*70)
    print(" ÖRNEK 1: Tek Ses Dosyası İçin XAI Açıklaması")
    print("="*70)

    # Model yolu - MFCC-CNN kullanıyoruz
    model_path = 'results/mfcc/mfcc_cnn_model.h5'

    # Sınıf isimlerini yükle
    results_path = 'results/mfcc/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    # XAI objesi oluştur - model_type='mfcc_cnn'
    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    # Örnek ses dosyası
    audio_path = 'dataset/dog_bark/100032-3-0-0.wav'

    if not os.path.exists(audio_path):
        print(f"❌ Ses dosyası bulunamadı: {audio_path}")
        print("\n💡 İpucu: Kendi ses dosyanızın yolunu girin")
        return

    # Kapsamlı açıklama
    xai.comprehensive_explanation(audio_path)


def example_multiple_files():
    """Birden fazla ses dosyası için açıklama"""

    print("\n" + "="*70)
    print(" ÖRNEK 2: Birden Fazla Ses Dosyası İçin XAI Açıklaması")
    print("="*70)

    # Model yolu - MFCC-CNN kullanıyoruz
    model_path = 'results/mfcc/mfcc_cnn_model.h5'

    # Sınıf isimlerini yükle
    results_path = 'results/mfcc/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    # XAI objesi oluştur - model_type='mfcc_cnn'
    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    # Her sınıftan birer örnek al
    audio_files = []
    for class_name in class_names:
        class_dir = f'dataset/{class_name}'
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            if files:
                audio_files.append(os.path.join(class_dir, files[0]))

    if not audio_files:
        print("❌ Hiç ses dosyası bulunamadı!")
        return

    print(f"\n📊 {len(audio_files)} ses dosyası analiz edilecek\n")

    # Her dosya için açıklama
    for i, audio_path in enumerate(audio_files, 1):
        print(f"\n{'='*70}")
        print(f" Dosya {i}/{len(audio_files)}: {os.path.basename(audio_path)}")
        print(f"{'='*70}")

        # Sadece temporal ve frequency analizi (hızlı)
        xai.explain_temporal_contribution(audio_path, segment_duration=0.5)
        xai.explain_frequency_contribution(audio_path)


def example_lime_only():
    """Sadece LIME açıklaması"""

    print("\n" + "="*70)
    print(" ÖRNEK 3: Sadece LIME Açıklaması")
    print("="*70)

    model_path = 'results/mfcc/mfcc_cnn_model.h5'

    results_path = 'results/mfcc/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    audio_path = 'dataset/dog_bark/100032-3-0-0.wav'

    if not os.path.exists(audio_path):
        print(f"❌ Ses dosyası bulunamadı: {audio_path}")
        return

    # Sadece LIME
    xai.explain_with_lime(audio_path, num_features=20)


def example_temporal_only():
    """Sadece zamansal katkı analizi"""

    print("\n" + "="*70)
    print(" ÖRNEK 4: Sadece Zamansal Katkı Analizi")
    print("="*70)

    model_path = 'results/mfcc/mfcc_cnn_model.h5'

    results_path = 'results/mfcc/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    audio_path = 'dataset/dog_bark/100032-3-0-0.wav'

    if not os.path.exists(audio_path):
        print(f"❌ Ses dosyası bulunamadı: {audio_path}")
        return

    # Zamansal analiz
    contributions = xai.explain_temporal_contribution(audio_path, segment_duration=0.5)

    print(f"\nSegment Katkı Skorları:")
    for i, contrib in enumerate(contributions):
        bar = '█' * int(contrib * 50)
        print(f"  Segment {i+1}: {contrib:.2%} {bar}")


def example_frequency_only():
    """Sadece frekans katkı analizi"""

    print("\n" + "="*70)
    print(" ÖRNEK 5: Sadece Frekans Katkı Analizi")
    print("="*70)

    model_path = 'results/mfcc/mfcc_cnn_model.h5'

    results_path = 'results/mfcc/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            class_names = results['class_names']
    else:
        class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                      'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
                      'siren', 'street_music']

    xai = AudioXAI(
        model_path=model_path,
        class_names=class_names,
        model_type='mfcc_cnn'
    )

    audio_path = 'dataset/dog_bark/100032-3-0-0.wav'

    if not os.path.exists(audio_path):
        print(f"❌ Ses dosyası bulunamadı: {audio_path}")
        return

    # Frekans analizi
    band_contributions = xai.explain_frequency_contribution(audio_path)

    print(f"\nFrekans Bandı Katkı Skorları:")
    for band_name, contrib in band_contributions:
        bar = '█' * int(contrib * 50)
        print(f"  {band_name:20s}: {contrib:.2%} {bar}")


def main():
    """Ana menü"""

    print("\n" + "="*70)
    print(" 🎯 XAI (Explainable AI) - Ses Sınıflandırma")
    print("="*70)
    print("\nLütfen çalıştırmak istediğiniz örneği seçin:")
    print("\n1. Tek ses dosyası için kapsamlı açıklama")
    print("2. Birden fazla dosya için açıklama")
    print("3. Sadece LIME açıklaması")
    print("4. Sadece zamansal katkı analizi")
    print("5. Sadece frekans katkı analizi")
    print("0. Çıkış")

    choice = input("\nSeçiminiz (0-5): ").strip()

    if choice == '1':
        example_single_explanation()
    elif choice == '2':
        example_multiple_files()
    elif choice == '3':
        example_lime_only()
    elif choice == '4':
        example_temporal_only()
    elif choice == '5':
        example_frequency_only()
    elif choice == '0':
        print("\n👋 Çıkış yapılıyor...")
    else:
        print("\n❌ Geçersiz seçim!")


if __name__ == '__main__':
    main()
