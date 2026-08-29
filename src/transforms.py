from torchvision import transforms

# Menggunakan mean dan standard deviation dari dataset ImageNet 
# (best practice untuk transfer learning menggunakan pre-trained models PyTorch)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transformasi untuk data Train
# Termasuk augmentasi data untuk mencegah overfitting (Data Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),                           # Resize ke ukuran standar model
    transforms.RandomHorizontalFlip(p=0.5),                  # Flip gambar secara horizontal (50% probabilitas)
    transforms.RandomRotation(degrees=15),                   # Rotasi acak hingga 15 derajat
    transforms.ColorJitter(brightness=0.2, contrast=0.2,     # Ubah warna/kecerahan secara acak
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),                                   # Konversi ke Tensor dan scale piksel ke [0, 1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # Normalisasi ImageNet
])

# Transformasi untuk data Validation dan Test
# Hanya resize dan normalisasi (TIDAK ADA augmentasi pada val/test set)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Transformasi Train V2 — augmentasi lebih agresif untuk fine-tuning
# Dirancang untuk membantu model membedakan material reflektif (glass vs metal)
# dan material transparan (plastic vs glass) dengan variasi warna/kontras lebih tinggi
train_transform_v2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),                     # Rotasi lebih besar
    transforms.ColorJitter(brightness=0.4, contrast=0.4,       # Variasi warna lebih tinggi
                           saturation=0.4, hue=0.15),
    transforms.RandomAutocontrast(p=0.3),                      # Bantu bedakan refleksi vs matte
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
