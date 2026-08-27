# ♻️ Waste Classifier — Deep Learning Image Classification

Klasifikasi otomatis jenis sampah (cardboard, glass, metal, paper, plastic, trash) menggunakan Convolutional Neural Network dan transfer learning dengan PyTorch. Project ini dibangun sebagai studi kasus end-to-end: mulai dari model CNN dari nol, transfer learning dengan pretrained ResNet18, hingga (rencana) deployment ke perangkat mobile.

## 📌 Ringkasan

| | |
|---|---|
| **Task** | Multi-class image classification (6 kelas) |
| **Dataset** | [TrashNet](https://github.com/garythung/trashnet) (2,527 gambar) |
| **Framework** | PyTorch |
| **Best Model** | ResNet18 (transfer learning, frozen backbone) |
| **Best Val Accuracy** | 78.36% |
| **Status** | 🚧 In Progress — Tahap 4 dari 8 |

## 🎯 Motivasi

Pemilahan sampah yang tepat adalah langkah dasar untuk sistem daur ulang yang efektif, namun sering kali dilakukan secara manual dan tidak konsisten. Project ini mengeksplorasi apakah deep learning dapat mengotomatisasi klasifikasi jenis sampah dari gambar, sebagai fondasi untuk aplikasi assistive sorting (misalnya di TPS, bank sampah, atau aplikasi mobile edukasi lingkungan).

## 🗂️ Struktur Project

```
waste-classifier/
├── data/
│   ├── raw/              # Dataset asli TrashNet (6 subfolder kelas)
│   ├── train/            # 70% — augmented saat training
│   ├── val/               # 15% — untuk monitoring selama training
│   └── test/              # 15% — untuk evaluasi final (belum pernah dilihat model)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── misclassified_examples.png
├── src/
│   ├── dataset.py          # Custom PyTorch Dataset class (WasteDataset)
│   ├── transforms.py       # Augmentation & preprocessing pipeline
│   ├── model.py             # Baseline CNN dari scratch (SimpleCNN)
│   ├── model_transfer.py    # ResNet18 dengan transfer learning
│   ├── train.py              # Training loop — baseline model
│   ├── train_transfer.py     # Training loop — transfer learning
│   └── evaluate.py           # Evaluasi, confusion matrix, error analysis
├── models/                   # Model checkpoints (.pth) — di-gitignore, lihat Releases
├── requirements.txt
└── README.md
```

## 📊 Dataset

Menggunakan [TrashNet](https://github.com/garythung/trashnet), berisi 2,527 gambar sampah dalam 6 kategori:

| Kelas | Deskripsi |
|---|---|
| `cardboard` | Kardus, kotak karton |
| `glass` | Botol kaca, pecahan kaca |
| `metal` | Kaleng, tutup logam |
| `paper` | Kertas, koran |
| `plastic` | Botol plastik, kemasan |
| `trash` | Sampah campuran/tidak terklasifikasi |

Dataset di-split secara **stratified** menjadi train (70%) / validation (15%) / test (15%), memastikan proporsi kelas konsisten di setiap split.

## 🧠 Metodologi

### 1. Baseline Model — CNN dari Scratch
CNN sederhana dengan 3 convolutional block (32→64→128 channel), dilatih dari nol tanpa pretrained weights, untuk membangun pemahaman fundamental sebelum menggunakan transfer learning.

**Hasil:** 62.53% validation accuracy setelah 10 epoch (kondisi *underfitting* — kapasitas model terbatas untuk menangkap fitur visual kompleks tanpa pretrained knowledge).

### 2. Transfer Learning — ResNet18
Menggunakan ResNet18 pretrained pada ImageNet (1.2 juta gambar), dengan backbone dibekukan (*frozen*) dan hanya melatih ulang classifier head.

**Hasil:** 78.36% validation accuracy (epoch 7) — peningkatan **+15.83 percentage points** dari baseline, sambil melatih **99.5% lebih sedikit parameter** (132.9K vs 25.8M).

| Metrik | Baseline CNN | ResNet18 (Frozen) | Δ |
|---|---:|---:|---:|
| Val Accuracy | 62.53% | **78.36%** | +15.83pp |
| Trainable Params | 25.8M | 132.9K | -99.5% |
| Training Time/Epoch | ~17s | ~10s | ~40% lebih cepat |

### 3. Evaluation & Error Analysis *(in progress)*
Analisis mendalam menggunakan test set terpisah: classification report per kelas (precision/recall/F1), confusion matrix, dan inspeksi visual terhadap kasus yang salah diprediksi — untuk mengidentifikasi pasangan kelas yang sering tertukar sebelum melakukan iterasi improvement lebih lanjut.

### Rencana Selanjutnya
- [ ] **Fine-tuning** — Unfreeze layer akhir backbone dengan discriminative learning rates
- [ ] **Explainability** — Grad-CAM untuk visualisasi fokus perhatian model
- [ ] **Deployment** — Demo interaktif via Streamlit/Gradio, deploy ke Hugging Face Spaces
- [ ] **Mobile Deployment** — Konversi model ke TFLite untuk on-device inference (Android)

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, torchvision
- **Data Processing:** Pillow, NumPy
- **Evaluation:** scikit-learn (classification metrics), seaborn/matplotlib (visualisasi)
- **Environment:** Python 3.x, virtualenv

## 🚀 Cara Menjalankan

### 1. Setup Environment
```bash
python -m venv venv-wasteclassifier
source venv-wasteclassifier/bin/activate  # Windows: venv-wasteclassifier\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
kaggle datasets download -d feyzazkefe/trashnet
# Ekstrak ke data/raw/
```

### 3. Preprocessing & Split Dataset
```bash
python src/data_split.py
```

### 4. Training

**Baseline model:**
```bash
python src/train.py
```

**Transfer learning (ResNet18):**
```bash
python src/train_transfer.py
```

### 5. Evaluasi
```bash
python src/evaluate.py
```
Output: `classification_report.txt`, `confusion_matrix.png`, dan contoh gambar yang salah diklasifikasikan di folder `notebooks/`.

## 📈 Hasil & Insight Utama

1. **Transfer learning memberikan improvement signifikan** (+15.83pp) dengan biaya komputasi jauh lebih rendah dibanding training dari scratch — memvalidasi pendekatan yang umum dipakai di industri untuk dataset berukuran kecil-menengah.
2. **Tidak ada indikasi overfitting** pada model transfer learning (gap train-val accuracy kecil), menunjukkan model masih memiliki ruang untuk pembelajaran lebih dalam melalui fine-tuning backbone.
3. Analisis error per kelas sedang dilakukan untuk mengidentifikasi pola kesalahan model secara sistematis, sebagai dasar strategi improvement yang terarah (bukan trial-and-error).

## 📄 Lisensi

MIT License — bebas digunakan untuk pembelajaran dan pengembangan lebih lanjut.

## 🙋 Tentang Project Ini

Project ini dibangun sebagai bagian dari proses belajar deep learning secara terstruktur (dari CNN dasar hingga transfer learning dan deployment), sekaligus sebagai portofolio yang menunjukkan pemahaman end-to-end machine learning workflow: data preparation, model development, evaluation, dan (rencana) production deployment.