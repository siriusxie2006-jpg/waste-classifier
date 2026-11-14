"""Shared model construction and strict loading for merged experiments."""
import hashlib
import math
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def build_model(head='linear', pretrained=False, architecture='resnet18'):
    if architecture not in ('resnet18', 'multiscale_fixed', 'multiscale_gated'):
        raise ValueError(f'Unknown architecture: {architecture}')
    if architecture != 'resnet18' and head != 'linear':
        raise ValueError('Multi-scale variants require the shared linear classifier')
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if head == 'linear':
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    elif head == 'mlp':
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.ReLU(),
                                 nn.Dropout(.4), nn.Linear(256, len(CLASSES)))
    elif head == 'residual_capacity':
        from capacity_control import ResidualCapacityHead
        classifier = nn.Linear(model.fc.in_features, len(CLASSES))
        model.fc = ResidualCapacityHead(classifier)
    else:
        raise ValueError(f'Unknown classification head: {head}')
    if architecture != 'resnet18':
        from multiscale_model import MultiScaleResNet18
        model = MultiScaleResNet18(model, adaptive=architecture == 'multiscale_gated')
    return model


def set_training_phase(model, scope='head'):
    if scope not in ('head', 'layer4', 'layer3_layer4'):
        raise ValueError(f'Unknown fine-tuning scope: {scope}')
    active = ['fc'] + ([] if scope == 'head' else ['layer4'])
    if scope == 'layer3_layer4':
        active.append('layer3')
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = name.split('.')[0] in active
    for name in active:
        getattr(model, name).train()


def build_optimizer(model, head_lr=.001, backbone_lr=.0001, weight_decay=.0001):
    """AdamW for the selected blocks; historical defaults preserve the old recipe."""
    if any(not math.isfinite(value) or value <= 0 for value in (head_lr, backbone_lr)):
        raise ValueError('Learning rates must be finite and positive')
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    groups = [{'params': list(model.fc.parameters()), 'lr': head_lr}]
    for block in ['layer3', 'layer4']:
        parameters = [parameter for parameter in getattr(model, block).parameters() if parameter.requires_grad]
        if parameters:
            groups.append({'params': parameters, 'lr': backbone_lr})
    selected = [parameter for group in groups for parameter in group['params']]
    if len({id(parameter) for parameter in selected}) != len(selected):
        raise ValueError('An optimizer parameter was included more than once')
    if {id(parameter) for parameter in selected} != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError('Optimizer groups differ from the trainable parameters')
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def training_class_weights(records, mode='none'):
    if mode == 'none':
        return None
    if mode != 'inverse':
        raise ValueError(mode)
    counts = torch.bincount(torch.tensor([int(r['label']) for r in records
                                         if r['split'] == 'train']), minlength=len(CLASSES))
    if (counts == 0).any():
        raise ValueError('Every class needs training examples for inverse weighting')
    return counts.sum() / (len(CLASSES) * counts.float())


def load_checkpoint(path, device='cpu', manifest_path=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('classes') != CLASSES:
        raise ValueError('Checkpoint class order differs from the project class order')
    config = checkpoint.get('config', {})
    if manifest_path is not None:
        expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if config.get('split_manifest_sha256') != expected:
            raise ValueError('Checkpoint has no current split fingerprint or uses another split')
    model = build_model(config.get('head', 'linear'), architecture=config.get('architecture', 'resnet18'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval(), checkpoint
