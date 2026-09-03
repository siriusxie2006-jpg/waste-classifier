import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import WasteDataset
from model_finetune import create_resnet18_finetune
from transforms import val_test_transform, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, overlay_heatmap


def load_model(checkpoint_path, device):
    """Load finetuned model dari checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint['num_classes']
    classes = checkpoint['classes']

    model = create_resnet18_finetune(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, classes


def get_original_image(img_path, size=(224, 224)):
    """Load gambar asli sebagai numpy array (untuk overlay)."""
    img = Image.open(img_path).convert('RGB').resize(size)
    return np.array(img)


def find_samples(dataset, model, device, classes, n_correct=4, n_wrong=4, seed=42):
    """
    Cari campuran sample benar dan salah, dengan prioritas pada
    confusion pairs glass->metal dan cardboard->paper.
    """
    random.seed(seed)

    correct_samples = []
    wrong_samples = []
    # Prioritas: cari kasus glass->metal dan cardboard->paper
    glass_metal = []
    cardboard_paper = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img_tensor, true_label = dataset[idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
            confidence = probs[0, pred_label].item()

            sample = {
                'idx': idx,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'img_path': dataset.samples[idx][0],
                'correct': true_label == pred_label
            }

            if true_label == pred_label:
                correct_samples.append(sample)
            else:
                wrong_samples.append(sample)
                # Cek confusion pairs spesifik
                true_name = classes[true_label]
                pred_name = classes[pred_label]
                if true_name == 'glass' and pred_name == 'metal':
                    glass_metal.append(sample)
                elif true_name == 'cardboard' and pred_name == 'paper':
                    cardboard_paper.append(sample)

    print(f"Total correct: {len(correct_samples)}, wrong: {len(wrong_samples)}")
    print(f"Glass→Metal errors: {len(glass_metal)}, Cardboard→Paper errors: {len(cardboard_paper)}")

    # Susun sample: prioritaskan confusion pairs untuk wrong samples
    selected_wrong = []

    # Ambil max 2 glass->metal
    for s in glass_metal[:2]:
        selected_wrong.append(s)

    # Ambil max 2 cardboard->paper
    for s in cardboard_paper[:2]:
        selected_wrong.append(s)

    # Sisanya dari wrong samples lain (hindari duplikat)
    selected_ids = {s['idx'] for s in selected_wrong}
    other_wrong = [s for s in wrong_samples if s['idx'] not in selected_ids]
    random.shuffle(other_wrong)
    remaining_needed = n_wrong - len(selected_wrong)
    if remaining_needed > 0:
        selected_wrong.extend(other_wrong[:remaining_needed])

    # Ambil correct samples dari berbagai kelas
    random.shuffle(correct_samples)
    # Coba ambil 1 dari tiap kelas yang berbeda
    seen_classes = set()
    selected_correct = []
    for s in correct_samples:
        if s['true_label'] not in seen_classes and len(selected_correct) < n_correct:
            selected_correct.append(s)
            seen_classes.add(s['true_label'])
    # Jika kurang, tambah dari sisa
    if len(selected_correct) < n_correct:
        for s in correct_samples:
            if s['idx'] not in {x['idx'] for x in selected_correct} and len(selected_correct) < n_correct:
                selected_correct.append(s)

    return selected_correct, selected_wrong


def main():
    CHECKPOINT_PATH = "models/resnet18_finetuned.pth"
    TEST_DIR = "data/test"
    OUTPUT_PATH = "notebooks/gradcam_visualization.png"

    # === Device ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # === Load model dan dataset ===
    model, classes = load_model(CHECKPOINT_PATH, device)
    test_dataset = WasteDataset(TEST_DIR, transform=val_test_transform)

    # === Cari samples ===
    correct_samples, wrong_samples = find_samples(
        test_dataset, model, device, classes, n_correct=4, n_wrong=4
    )
    all_samples = correct_samples + wrong_samples
    print(f"\nTotal samples untuk visualisasi: {len(all_samples)}")

    # === Setup Grad-CAM (target: layer4, block terakhir ResNet) ===
    gradcam = GradCAM(model, target_layer=model.layer4)

    # === Generate visualisasi ===
    n = len(all_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

    for i, sample in enumerate(all_samples):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]

        # Load gambar
        img_path = sample['img_path']
        original_img = get_original_image(img_path)

        # Forward + Grad-CAM
        img_tensor = test_dataset[sample['idx']][0].unsqueeze(0).to(device)
        cam = gradcam.generate(img_tensor, class_idx=sample['pred_label'])

        # Overlay heatmap
        overlaid = overlay_heatmap(original_img, cam, alpha=0.5)

        # Tampilkan
        ax.imshow(overlaid)

        true_name = classes[sample['true_label']]
        pred_name = classes[sample['pred_label']]
        conf = sample['confidence'] * 100

        if sample['correct']:
            # Prediksi benar: hijau
            title_color = 'green'
            title = f"✓ {true_name}\nConf: {conf:.1f}%"
        else:
            # Prediksi salah: merah
            title_color = 'red'
            title = f"✗ {true_name} → {pred_name}\nConf: {conf:.1f}%"

        ax.set_title(title, fontsize=11, color=title_color, fontweight='bold')
        ax.axis('off')

        # Border warna
        border_color = '#2ecc71' if sample['correct'] else '#e74c3c'
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3)

    # Sembunyikan subplot kosong
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis('off')

    fig.suptitle('Grad-CAM Visualization — ResNet18 Fine-tuned\n(Baris 1: Prediksi Benar | Baris 2: Prediksi Salah)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualisasi disimpan ke: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
