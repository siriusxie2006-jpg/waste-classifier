import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk save langsung ke file
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Tambahkan src/ ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import WasteDataset
from model_transfer import create_resnet18_model
from transforms import val_test_transform, IMAGENET_MEAN, IMAGENET_STD


def load_model(checkpoint_path, device):
    """Load model terbaik dari checkpoint. Mendukung frozen dan finetuned model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    classes = checkpoint['classes']

    # Deteksi tipe model dari checkpoint
    model_type = checkpoint.get('model_type', 'resnet18_frozen')
    if model_type == 'resnet18_finetuned':
        from model_finetune import create_resnet18_finetune
        model = create_resnet18_finetune(num_classes=num_classes)
    else:
        model = create_resnet18_model(num_classes=num_classes, freeze_backbone=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model: {model_type}")
    return model, classes, checkpoint['class_to_idx']


def get_predictions(model, dataloader, device):
    """Jalankan inference pada seluruh dataset, kumpulkan prediksi dan probabilitas."""
    all_labels = []
    all_preds = []
    all_probs = []
    all_indices = []

    idx = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # Konversi logits ke probabilitas

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_indices.extend(range(idx, idx + len(labels)))
            idx += len(labels)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_indices)
    )


def save_classification_report(y_true, y_pred, classes, output_path):
    """Generate dan simpan classification report ke file."""
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open(output_path, 'w') as f:
        f.write("Classification Report - Waste Classifier (ResNet18 Frozen)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test set evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(report)
    return report


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Visualisasikan confusion matrix sebagai heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor='gray'
    )
    plt.title('Confusion Matrix — ResNet18 (Frozen Backbone)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix disimpan ke: {output_path}")
    return cm


def find_top_confusion_pairs(cm, classes, top_n=5):
    """Cari pasangan kelas yang paling sering tertukar (off-diagonal)."""
    pairs = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if i != j:  # Hanya ambil off-diagonal (salah prediksi)
                pairs.append((classes[i], classes[j], cm[i][j]))

    # Urutkan berdasarkan jumlah error terbanyak
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {top_n} Confusion Pairs (Actual → Predicted):")
    print("-" * 50)
    for actual, predicted, count in pairs[:top_n]:
        print(f"  {actual:>10} → {predicted:<10}: {count} gambar")

    return pairs[:top_n]


def denormalize(tensor):
    """Kembalikan tensor yang sudah dinormalisasi ke range [0, 1] untuk visualisasi."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def plot_misclassified(dataset, y_true, y_pred, y_probs, classes, output_path, n=9):
    """Visualisasikan grid contoh gambar yang salah diprediksi."""
    # Cari indeks gambar yang salah prediksi
    wrong_indices = np.where(y_true != y_pred)[0]

    if len(wrong_indices) == 0:
        print("Tidak ada gambar yang salah diprediksi; visualisasi dilewati.")
        return

    if len(wrong_indices) < n:
        n = len(wrong_indices)
        print(f"Hanya {n} gambar yang salah diprediksi, menampilkan semua.")

    # Pilih n sampel acak dari yang salah
    random.seed(42)
    selected = random.sample(list(wrong_indices), n)

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    fig.suptitle('Contoh Gambar yang Salah Diprediksi', fontsize=16, y=1.02)

    for i, idx in enumerate(selected):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]

        # Load gambar asli (tanpa transform) untuk visualisasi yang lebih jelas
        img_path, _ = dataset.samples[idx]
        img = Image.open(img_path).convert('RGB')

        true_label = classes[y_true[idx]]
        pred_label = classes[y_pred[idx]]
        confidence = y_probs[idx][y_pred[idx]] * 100  # Confidence untuk kelas yang diprediksi

        ax.imshow(img)
        ax.set_title(
            f"Asli: {true_label}\nPrediksi: {pred_label} ({confidence:.1f}%)",
            fontsize=10, color='red', fontweight='bold'
        )
        ax.axis('off')

    # Sembunyikan subplot kosong jika ada
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Misclassified examples disimpan ke: {output_path}")


def main():
    # Terima checkpoint path dari CLI, atau default ke resnet18_frozen.pth
    CHECKPOINT_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/resnet18_frozen.pth"
    TEST_DIR = "data/test"
    OUTPUT_DIR = "notebooks"

    # === Auto-detect device ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Menggunakan device: {device}")

    # === Load model dan dataset ===
    model, classes, class_to_idx = load_model(CHECKPOINT_PATH, device)
    test_dataset = WasteDataset(TEST_DIR, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Test set: {len(test_dataset)} gambar")
    print(f"Kelas: {classes}\n")

    # === Jalankan inference ===
    y_true, y_pred, y_probs, indices = get_predictions(model, test_loader, device)

    # === Overall accuracy ===
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    test_acc = correct / total
    wrong = total - correct
    print(f"Overall Test Accuracy: {test_acc:.2%} ({correct}/{total})")
    print(f"Total gambar salah prediksi: {wrong} ({wrong/total:.2%})\n")

    # === Classification report ===
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    save_classification_report(y_true, y_pred, classes, report_path)
    print(f"\nClassification report disimpan ke: {report_path}")

    # === Confusion matrix ===
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    cm = plot_confusion_matrix(y_true, y_pred, classes, cm_path)

    # === Top confusion pairs ===
    find_top_confusion_pairs(cm, classes, top_n=5)

    # === Misclassified examples ===
    misc_path = os.path.join(OUTPUT_DIR, "misclassified_examples.png")
    plot_misclassified(test_dataset, y_true, y_pred, y_probs, classes, misc_path, n=9)

    # === Simpan raw predictions untuk analisis lanjutan ===
    raw_path = os.path.join(OUTPUT_DIR, "raw_predictions.npz")
    np.savez(
        raw_path,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs,
        classes=np.array(classes)
    )
    print(f"Raw predictions disimpan ke: {raw_path}")


if __name__ == "__main__":
    main()
