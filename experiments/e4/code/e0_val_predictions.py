"""Dump E0 validation predictions/probabilities (the reference arm for E4).

Only Validation is read here. Test is never touched.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C


def main() -> int:
    out_dir = os.path.join(C.E4_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C.load_e0(device)
    model.eval()
    C.freeze_all_bn(model)

    samples = C.list_split("val")
    ds = C.ImageListDataset(samples, C.eval_transform(True))
    loader = torch.utils.data.DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)

    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    print(f"E0 val: n={len(labels)} acc={acc:.6f} correct={(preds == labels).sum()}")

    np.savez(
        os.path.join(out_dir, "e0_val_probs.npz"),
        probs=probs,
        labels=labels,
        preds=preds,
        rel=np.array([s.rel for s in samples]),
    )
    C.dump_json(
        os.path.join(out_dir, "e0_val_summary.json"),
        {
            "n": int(len(labels)),
            "accuracy": acc,
            "n_correct": int((preds == labels).sum()),
            "metrics": C.summary_metrics(labels, preds),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
