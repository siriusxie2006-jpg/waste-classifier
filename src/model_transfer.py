import torch.nn as nn
from torchvision import models


def create_resnet18_model(num_classes, freeze_backbone=True):
    """
    Load ResNet18 pre-trained pada ImageNet dan ganti classifier-nya.

    Args:
        num_classes (int): Jumlah kelas output
        freeze_backbone (bool): Jika True, bekukan semua layer kecuali fc layer

    Returns:
        model: ResNet18 yang sudah dimodifikasi
    """
    # Load ResNet18 dengan weights ImageNet terbaru
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Bekukan semua parameter backbone jika diminta
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Ganti fc layer terakhir dengan custom classifier
    # ResNet18 fc layer asli: Linear(512, 1000)
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    # Layer baru otomatis requires_grad=True, jadi tidak perlu di-unfreeze manual

    return model
