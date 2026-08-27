import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Tambahkan src/ ke path agar bisa import modul lokal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import WasteDataset
from model import SimpleCNN
from transforms import train_transform, val_test_transform


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Jalankan satu epoch training, return (avg_loss, accuracy)."""
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

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Jalankan evaluasi pada validation set, return (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Matikan gradient computation untuk efisiensi
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # === Hyperparameter ===
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    CHECKPOINT_PATH = "models/baseline_cnn.pth"

    # === Auto-detect device ===
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Menggunakan device: {device}")

    # === Load Dataset ===
    train_dataset = WasteDataset("data/train", transform=train_transform)
    val_dataset = WasteDataset("data/val", transform=val_test_transform)

    print(f"Jumlah kelas: {len(train_dataset.classes)} -> {train_dataset.classes}")
    print(f"Train set: {len(train_dataset)} gambar")
    print(f"Val set:   {len(val_dataset)} gambar")

    # === DataLoader ===
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # === Model, Loss, Optimizer ===
    num_classes = len(train_dataset.classes)
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Hitung total parameter model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameter: {total_params:,} | Trainable: {trainable_params:,}")

    # === Training Loop ===
    print(f"\n{'='*70}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Time':>6}")
    print(f"{'='*70}")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>10.4f} | {val_acc:>8.2%} | {elapsed:>5.1f}s")

    print(f"{'='*70}")

    # === Simpan model checkpoint ===
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes,
    }, CHECKPOINT_PATH)
    print(f"\nModel checkpoint disimpan ke: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
