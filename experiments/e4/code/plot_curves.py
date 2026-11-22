"""Draw training curves with Pillow (no matplotlib in this environment)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

RUNS = os.path.join(C.E4_ROOT, "runs", "e4_2")
OUT = os.path.join(C.E4_ROOT, "results", "figures")
COLORS = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]


def collect():
    groups: dict[str, list[list[dict]]] = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        hp = os.path.join(d, "EPOCH_HISTORY.json")
        cp = os.path.join(d, "RUN_CONFIG.json")
        if not (os.path.exists(hp) and os.path.exists(cp)) or "_smoke" in d:
            continue
        cfg = json.load(open(cp, encoding="utf-8"))
        hist = json.load(open(hp, encoding="utf-8"))
        if len(hist) < cfg.get("epochs", 10):
            continue
        key = f"{cfg['unfreeze']}/{cfg['preprocess']}"
        groups.setdefault(key, []).append(hist)
    return groups


def draw_curve(groups, field, title, ylo, yhi, fname):
    W, H, M = 1100, 720, 90
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([M, M, W - 40, H - M], outline=(0, 0, 0))
    for i in range(11):
        y = H - M - (H - 2 * M) * i / 10
        d.line([M, y, W - 40, y], fill=(230, 230, 230))
        val = ylo + (yhi - ylo) * i / 10
        d.text((18, y - 6), f"{val:.2f}", fill=(0, 0, 0))
    for e in range(1, 11):
        x = M + (W - 40 - M) * (e - 1) / 9
        d.line([x, H - M, x, H - M + 5], fill=(0, 0, 0))
        d.text((x - 4, H - M + 8), str(e), fill=(0, 0, 0))
    d.text((W // 2 - 120, 30), title, fill=(0, 0, 0))
    d.text((W // 2 - 60, H - 40), "epoch", fill=(0, 0, 0))

    def to_xy(e, v):
        x = M + (W - 40 - M) * (e - 1) / 9
        v = max(ylo, min(yhi, v))
        y = H - M - (H - 2 * M) * (v - ylo) / (yhi - ylo)
        return x, y

    for gi, (key, runs) in enumerate(sorted(groups.items())):
        col = COLORS[gi % len(COLORS)]
        arr = np.array([[r[field] for r in run] for run in runs], dtype=float)
        mean = arr.mean(0)
        if len(runs) > 1:
            lo, hi = arr.min(0), arr.max(0)
            d.line([to_xy(e + 1, hi[e]) for e in range(10)], fill=col, width=1)
            d.line([to_xy(e + 1, lo[e]) for e in range(10)], fill=col, width=1)
        d.line([to_xy(e + 1, mean[e]) for e in range(10)], fill=col, width=3)
        for e in range(10):
            x, y = to_xy(e + 1, mean[e])
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
        lx, ly = M + 20, M + 20 + gi * 20
        d.rectangle([lx, ly, lx + 14, ly + 10], fill=col)
        d.text((lx + 20, ly - 2), f"{key} (mean of {len(runs)} seeds)", fill=(0, 0, 0))
    path = os.path.join(OUT, fname)
    img.save(path)
    print("wrote", path)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    groups = collect()
    if not groups:
        print("no runs")
        return 0
    print("groups:", {k: len(v) for k, v in groups.items()})
    draw_curve(groups, "val_macro_f1", "Validation macro-F1 by epoch", 0.68, 0.92, "curve_val_macro_f1.png")
    draw_curve(groups, "val_accuracy", "Validation accuracy by epoch", 0.68, 0.95, "curve_val_accuracy.png")
    draw_curve(groups, "train_accuracy", "Train accuracy by epoch", 0.65, 1.0, "curve_train_accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
