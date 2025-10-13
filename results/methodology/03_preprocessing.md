# Ön-İşleme Metodolojisi

## Veri Kaynağı ve Segmentasyon
Kayıtlar 16000 Hz örnekleme ile yüklendi; sessiz kısımlar kırpılarak 3.0 saniyelik segmentlere ayrıldı ve kanal başına Z-score ile normalize edildi.

## Parça-Temelli Bölme
Track kimliği baz alınarak segmentler birden fazla split'e düşmeyecek şekilde %80 eğitim, %10 doğrulama ve %10 test oranlarında ayrıldı.

## Öznitelik Çıkarımı
n_fft=1024, hop_length=256, win_length=1024 parametreleriyle STFT uygulandı; Mel (n_mels=128, fmin=20 Hz, fmax=8000 Hz) spektrogramlarının yanı sıra MFCC, harmonic-percussive ayrımı, Chroma ve Tempogram temsilleri hesaplandı.

## Augmentasyon Protokolü (Plan)
SNR hedefleri +5 dB, +0 dB, -5 dB, -10 dB; time-stretch oranları 0.9–1.1; pitch-shift adımları -2–2 yarım ses olarak planlandı.

## Kalite/Kontrol
Sınıf dağılımı (sınıflar: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music) izlenerek dengesizlikte model dışı stratejiler (sınıf ağırlıklandırma / örnekleme) değerlendirilir.
