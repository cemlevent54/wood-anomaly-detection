import os
import cv2
import numpy as np
from tqdm import tqdm

# Dataset yolu
dataset_path = "wood_dataset/wood"  # Mevcut dataset yolu
output_dataset_path = "wood_dataset/processed_wood_dataset"  # Çıkış dizini

# İşlenecek dizinler
directories = ["ground_truth/defect", "test/defect", "test/good", "train/good"]

# Çıkış boyutu
TARGET_SIZE = (256, 256)

# Görüntü işleme fonksiyonu
def preprocess_image(image_path):
    # Görüntüyü yükle (RGB)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Boyutlandırma (256x256)
    img_resized = cv2.resize(img, TARGET_SIZE)

    # Normalizasyon (0-1 aralığı)
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized

# Dataseti işle
for directory in directories:
    dir_path = os.path.join(dataset_path, directory)  # Girdi dizini
    output_dir_path = os.path.join(output_dataset_path, directory)  # Çıkış dizini

    # Çıkış dizinini oluştur
    os.makedirs(output_dir_path, exist_ok=True)

    # Girdi dizini yoksa, devam etme
    if not os.path.exists(dir_path):
        continue

    for filename in tqdm(os.listdir(dir_path), desc=f"Processing {directory}"):
        file_path = os.path.join(dir_path, filename)  # Girdi dosyasının yolu
        output_file_path = os.path.join(output_dir_path, os.path.splitext(filename)[0] + ".jpg")  # Çıktı dosyasının yolu

        # Görüntü formatı kontrolü (jpg, png, jpeg)
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # Görüntüyü işle
            processed_img = preprocess_image(file_path)

            # İşlenmiş görüntüyü kaydet (Yeni dizine .jpg formatında)
            cv2.imwrite(output_file_path, (processed_img * 255).astype(np.uint8))

print(f"Tüm resimler başarıyla '{output_dataset_path}' dizinine işlendi ve kaydedildi.")
