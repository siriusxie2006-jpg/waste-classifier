import torch.nn as nn
from torchvision import models


def create_resnet18_finetune(num_classes):
    """
    Load ResNet18 pre-trained dan unfreeze layer3, layer4, dan fc untuk fine-tuning.
    
    Strategi: Hanya 2 block terakhir backbone yang di-finetune, sehingga
    fitur low-level (edges, textures) dari layer1-2 tetap frozen,
    sementara fitur high-level bisa diadaptasi ke domain waste images.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze semua parameter terlebih dahulu
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3 dan layer4 (2 block terakhir backbone)
    for name, param in model.named_parameters():
        if name.startswith('layer3') or name.startswith('layer4'):
            param.requires_grad = True

    # Ganti fc layer dengan custom classifier (otomatis requires_grad=True)
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model
