"""E4-4: validation-time ensemble and horizontal-flip TTA.

Members are fine-tuned checkpoints produced by e4_2_finetune.py. Every member is
evaluated on Validation only; Test is never read here.

Usage:
  python e4_4_ensemble_tta.py --members runA/checkpoint_epoch10.pth runB/... \
      --tta hflip --name l4_s42_s43_hflip
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

OUT_DIR = os.path.join(C.E4_ROOT, "results", "e4_4")


def hflip_eval_transform(preprocess: str):
    base = C.eval_transform(preprocess == "resize")
    return transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + list(base.transforms))


def load_member(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    model = C.make_resnet18_e0()
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    C.freeze_all_bn(model)
    return model, ck


@torch.no_grad()
def predict(model, loader, device):
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs.append(F.softmax(model(x), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--preprocess", default="resize", choices=["resize", "pad"])
    ap.add_argument("--tta", default="none", choices=["none", "hflip"])
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_samples = C.list_split("val")
    tf = C.eval_transform(args.preprocess == "resize")
    ds = C.ImageListDataset(val_samples, tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    ds_flip = C.ImageListDataset(val_samples, hflip_eval_transform(args.preprocess)) if args.tta == "hflip" else None
    dl_flip = torch.utils.data.DataLoader(ds_flip, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0) if ds_flip else None

    member_probs = []
    member_info = []
    labels = None
    for m in args.members:
        model, ck = load_member(m, device)
        p, y = predict(model, dl, device)
        labels = y if labels is None else labels
        assert np.array_equal(labels, y)
        if dl_flip is not None:
            pf, _ = predict(model, dl_flip, device)
            p = 0.5 * (p + pf)
        member_probs.append(p)
        metrics = C.summary_metrics(labels, p.argmax(1))
        member_info.append({
            "checkpoint": m,
            "sha256": C.sha256_file(m),
            "unfreeze": ck.get("unfreeze"),
            "preprocess": ck.get("preprocess"),
            "seed": ck.get("seed"),
            "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "n_correct": metrics["n_correct"],
        })
        print(f"member {os.path.basename(os.path.dirname(m))}: mF1={metrics['macro_f1']:.4f} "
              f"acc={metrics['accuracy']:.4f} correct={metrics['n_correct']}", flush=True)

    mean_prob = np.mean(member_probs, axis=0)
    ens = C.summary_metrics(labels, mean_prob.argmax(1))

    # pairwise prediction agreement (complementarity check)
    preds = [p.argmax(1) for p in member_probs]
    agree = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            agree.append(float((preds[i] == preds[j]).mean()))

    out = {
        "name": args.name,
        "preprocess": args.preprocess,
        "tta": args.tta,
        "n_members": len(args.members),
        "members": member_info,
        "pairwise_agreement": agree,
        "ensemble": {
            "val_accuracy": ens["accuracy"],
            "val_macro_f1": ens["macro_f1"],
            "val_weighted_f1": ens["weighted_f1"],
            "n_correct": ens["n_correct"],
            "per_class": ens["per_class"],
            "confusion_matrix": ens["confusion_matrix"],
        },
    }
    e0 = np.load(os.path.join(C.E4_ROOT, "results", "e0_val_probs.npz"), allow_pickle=True)
    e0_preds = e0["preds"]
    y = e0["labels"]
    assert np.array_equal(y, labels)
    out["ensemble"]["delta_vs_e0_correct"] = int(ens["n_correct"] - (e0_preds == y).sum())
    out["ensemble"]["mcnemar_vs_e0"] = C.paired_mcnemar(e0_preds, mean_prob.argmax(1), y)
    out["e0_val_n_correct"] = int((e0_preds == y).sum())

    C.dump_json(os.path.join(OUT_DIR, f"{args.name}.json"), out)
    np.savez(os.path.join(OUT_DIR, f"{args.name}_probs.npz"),
             labels=labels, ensemble_probs=mean_prob, preds=mean_prob.argmax(1))
    print(f"ENSEMBLE {args.name}: acc={ens['accuracy']:.4f} mF1={ens['macro_f1']:.4f} "
          f"correct={ens['n_correct']} (E0={out['e0_val_n_correct']}) "
          f"p={out['ensemble']['mcnemar_vs_e0']['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
