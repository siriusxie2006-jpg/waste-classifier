"""E4 common utilities: data, model, transforms, metrics.

Project: DS4023 Robust Waste Classification Under Image Degradation.
Base checkpoint: E0 frozen ImageNet ResNet18, head 512->256->ReLU->Dropout(0.4)->6.

All experiments in E4_FINAL import from here so that the protocol is identical
everywhere: same split, same class order, same augmentation for fine-tuning,
same metric definitions.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------- constants
PROJECT_ROOT = r"D:\ml_project"
DATA_ROOT = os.path.join(PROJECT_ROOT, "third_party", "waste-classifier", "data")
E4_ROOT = os.path.join(PROJECT_ROOT, "E4_FINAL")
E0_CKPT = os.path.join(PROJECT_ROOT, "E0_GROUPED_BASELINE", "models", "resnet18_frozen_best.pth")
SPLIT_MANIFEST = os.path.join(DATA_ROOT, "manifests", "split_manifest.csv")

CLASSES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
NUM_CLASSES = len(CLASSES)
SEED_DEFAULT = 42
EPOCHS_DEFAULT = 10
BATCH_SIZE = 32
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- transforms
def train_transform_e0() -> transforms.Compose:
    """Exactly the E0 training augmentation (src/transforms.py: train_transform)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def eval_transform(resize: bool = True) -> transforms.Compose:
    if resize:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        pad_to_square_224(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class pad_to_square_224:
    """Aspect-preserving resize into 224x224 then pad with ImageNet mean colour."""

    def __init__(self, size: int = 224):
        self.size = size
        # ImageNet mean in 0..255 RGB -> a neutral grey that normalises to ~0.
        self.fill = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def train_transform_pad() -> transforms.Compose:
    return transforms.Compose([
        pad_to_square_224(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------- dataset
@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    rel: str


def list_split(split: str) -> list[Sample]:
    """List a split from the on-disk class folders (same order as E0 loaders)."""
    root = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    out: list[Sample] = []
    for ci, cls in enumerate(CLASSES):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for name in sorted(os.listdir(cdir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                out.append(Sample(os.path.join(cdir, name), ci, f"{cls}/{name}"))
    return out


class ImageListDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], tf):
        self.samples = list(samples)
        self.tf = tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        return self.tf(img), s.label


# ---------------------------------------------------------------- model
def make_resnet18_e0(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def load_e0(device: torch.device, num_classes: int = NUM_CLASSES) -> nn.Module:
    ck = torch.load(E0_CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model = make_resnet18_e0(num_classes)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return model


def freeze_all_bn(model: nn.Module) -> None:
    """BN modules stay in eval mode and never update running stats or affine params."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def configure_trainable(model: nn.Module, unfreeze: str) -> tuple[list, list]:
    """Return (backbone_params, head_params) that require grad.

    unfreeze:
        'none'      -> E0 behaviour: head only
        'layer4'    -> layer4.* + head
        'layer3+4'  -> layer3.* + layer4.* + head
    """
    unfreeze = unfreeze.lower()
    allowed_prefixes: tuple[str, ...]
    if unfreeze in ("none", "head", "frozen"):
        allowed_prefixes = ()
    elif unfreeze == "layer4":
        allowed_prefixes = ("layer4.",)
    elif unfreeze in ("layer3+4", "layer34", "layer3_4"):
        allowed_prefixes = ("layer3.", "layer4.")
    else:
        raise ValueError(f"unknown unfreeze mode: {unfreeze}")

    for name, p in model.named_parameters():
        is_head = name.startswith("fc.")
        is_backbone_allowed = any(name.startswith(pref) for pref in allowed_prefixes)
        p.requires_grad = bool(is_head or is_backbone_allowed)

    freeze_all_bn(model)

    backbone = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc.")]
    return backbone, head


def train_mode_keep_bn_eval(model: nn.Module) -> None:
    model.train()
    freeze_all_bn(model)


# ---------------------------------------------------------------- metrics
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_CLASSES) -> dict:
    cm = confusion_matrix(y_true, y_pred, k)
    out = {}
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[name] = {
            "support": int(cm[i, :].sum()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return out


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    per = per_class_report(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(np.mean([v["f1"] for v in per.values()]))
    support = np.array([v["support"] for v in per.values()], dtype=np.float64)
    wf1 = float(np.sum(support * np.array([v["f1"] for v in per.values()])) / support.sum())
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": wf1,
        "per_class": per,
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
    }


def mcnemar_exact(b: int, c: int) -> dict:
    """Exact binomial McNemar (two-sided) on discordant counts b, c."""
    from math import comb

    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "n_discordant": 0}
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": float(min(1.0, 2 * tail)), "n_discordant": n}


def paired_mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict:
    a_ok = pred_a == y
    b_ok = pred_b == y
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    res = mcnemar_exact(b, c)
    res["delta_correct"] = c - b
    return res


# ---------------------------------------------------------------- io helpers
def set_determinism(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
