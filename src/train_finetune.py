import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import WasteDataset
from model_finetune import create_resnet18_finetune
from transforms import train_transform_v2, val_test_transform


def compute_class_weights(dataset):
    """
    Hitung class weights berbasis inverse frequency.
    Kelas minoritas (misal trash) mendapat weight lebih tinggi agar
    loss function memberi penalti lebih besar saat kelas tersebut salah.
    """
    label_counts = Counter()
    for _, label in dataset.samples:
        label_counts[label] += 1

    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # Inverse frequency: weight_i = total / (num_classes * count_i)
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        weights[cls_idx] = total / (num_classes * label_counts[cls_idx])

    print("Class weights (inverse frequency):")
    for cls_name, cls_idx in sorted(dataset.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {cls_name}: {weights[cls_idx]:.4f} (n={label_counts[cls_idx]})")

    return weights


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Jalankan satu epoch training."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Evaluasi pada validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, correct / total


def main():
    # === Hyperparameter ===
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    CHECKPOINT_PATH = "models/resnet18_finetuned.pth"

    # Discriminative learning rates
    LR_BACKBONE = 1e-4   # layer3 & layer4 — belajar lebih pelan
    LR_HEAD = 1e-3        # fc layer — belajar lebih cepat

    # === Auto-detect device ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Menggunakan device: {device}")

    # === Load Dataset ===
    train_dataset = WasteDataset("data/train", transform=train_transform_v2)
    val_dataset = WasteDataset("data/val", transform=val_test_transform)

    print(f"Jumlah kelas: {len(train_dataset.classes)} -> {train_dataset.classes}")
    print(f"Train set: {len(train_dataset)} gambar")
    print(f"Val set:   {len(val_dataset)} gambar\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # === Model ===
    num_classes = len(train_dataset.classes)
    model = create_resnet18_finetune(num_classes=num_classes).to(device)

    # Hitung parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameter: {total_params:,} | Trainable: {trainable:,} | Frozen: {total_params - trainable:,}")

    # === Class weights untuk mengatasi imbalance (terutama kelas trash) ===
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # === Discriminative Learning Rates ===
    # Pisahkan parameter ke 2 grup: backbone (layer3+4) dan head (fc)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('layer3') or name.startswith('layer4'):
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},  # Backbone: lr kecil
        {'params': head_params, 'lr': LR_HEAD}            # Head: lr besar
    ])

    # === LR Scheduler: kurangi lr jika val_loss stagnan ===
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # === Training Loop ===
    best_val_acc = 0.0
    best_model_state = None

    print(f"\n{'='*85}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR(bb)':>9} | {'LR(fc)':>9} | {'Best':>4}")
    print(f"{'='*85}")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler berdasarkan val_loss
        scheduler.step(val_loss)

        # Ambil current learning rate dari tiap parameter group
        lr_bb = optimizer.param_groups[0]['lr']
        lr_fc = optimizer.param_groups[1]['lr']

        elapsed = time.time() - start_time

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        marker = " *" if is_best else ""
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>10.4f} | {val_acc:>8.2%} | {lr_bb:>9.6f} | {lr_fc:>9.6f} |{marker}")

    print(f"{'='*85}")
    print(f"\nBest Val Accuracy: {best_val_acc:.2%}")

    # === Simpan best model ===
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes,
        'best_val_acc': best_val_acc,
        'model_type': 'resnet18_finetuned',
    }, CHECKPOINT_PATH)
    print(f"Best model checkpoint disimpan ke: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
