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
    if not freeze_backbone:
        raise ValueError("ResNet18 Frozen hanya mendukung backbone yang sepenuhnya dibekukan.")

    # Load ResNet18 dengan weights ImageNet terbaru
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the complete feature extractor; the replacement head remains trainable.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")

    # Ganti fc layer terakhir dengan custom classifier
    # ResNet18 fc layer asli: Linear(512, 1000)
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    # Keep the backbone in eval mode even when the training loop switches modes.
    set_frozen_resnet18_mode(model, training=True)

    assert_frozen_resnet18_parameters(model)

    return model


def set_frozen_resnet18_mode(model, training):
    """Set train/eval mode while keeping every backbone module in eval mode."""
    model.train(training)
    for name, module in model.named_children():
        module.train(training if name == "fc" else False)

    for name, module in model.named_modules():
        if not name:
            continue
        expected_training = training if name == "fc" or name.startswith("fc.") else False
        if module.training != expected_training:
            raise RuntimeError(f"Unexpected training mode for ResNet18 module {name}.")


def assert_frozen_resnet18_parameters(model):
    """Fail early unless only the classification head has trainable parameters."""
    backbone_parameters = [
        param for name, param in model.named_parameters()
        if not name.startswith("fc.")
    ]
    head_parameters = list(model.fc.parameters())

    if not backbone_parameters or any(param.requires_grad for param in backbone_parameters):
        raise RuntimeError("ResNet18 backbone parameters must all have requires_grad=False.")
    if not head_parameters or not all(param.requires_grad for param in head_parameters):
        raise RuntimeError("All classification head parameters must be trainable.")


def snapshot_batchnorm_running_stats(model):
    """Copy BatchNorm running statistics for the training invariant check."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            stats[name] = (
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
            )
    return stats


def assert_batchnorm_running_stats_unchanged(model, before):
    """Raise if a BatchNorm running mean or variance changed during an epoch."""
    for name, module in model.named_modules():
        if name not in before:
            continue
        mean_before, var_before = before[name]
        if mean_before is not None and not module.running_mean.equal(mean_before):
            raise RuntimeError(f"BatchNorm running_mean changed in backbone module {name}.")
        if var_before is not None and not module.running_var.equal(var_before):
            raise RuntimeError(f"BatchNorm running_var changed in backbone module {name}.")
