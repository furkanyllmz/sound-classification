import pandas as pd
import shutil
import os

# CSV dosyasını oku
df = pd.read_csv('archive/UrbanSound8K.csv')

# air_conditioner sınıfına ait dosyaları filtrele
air_conditioner_files = df[df['class'] == 'dog_bark']

# Hedef klasör
target_dir = 'dataset/dog_bark'
os.makedirs(target_dir, exist_ok=True)

# Dosyaları taşı
moved_count = 0
for _, row in air_conditioner_files.iterrows():
    filename = row['slice_file_name']
    fold = row['fold']
    source_path = f'archive/fold{fold}/{filename}'
    target_path = f'{target_dir}/{filename}'

    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
        moved_count += 1
        print(f'Taşındı: {filename}')
    else:
        print(f'Bulunamadı: {source_path}')

print(f'\nToplam {moved_count} dosya taşındı.')
