"""E4-2 / E4-3: fine-tune the E0 checkpoint with a partially unfrozen backbone.

Fixed protocol (per task spec):
  * start from the E0 checkpoint,
  * BN layers always in eval mode, BN affine params frozen, running stats frozen,
  * backbone lr 1e-4, head lr 1e-3, Adam, batch 32,
  * 10 epochs, E0 augmentation, no auxiliary loss, no class weights.

Variants
  --unfreeze {layer4,layer3+4}   E4-2 arm A / arm B
  --preprocess {resize,pad}      E4-3 arm A / arm B

Test is never touched.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C


def build_loaders(seed: int, preprocess: str):
    tf_train = C.train_transform_e0() if preprocess == "resize" else C.train_transform_pad()
    tf_eval = C.eval_transform(preprocess == "resize")

    train_samples = C.list_split("train")
    val_samples = C.list_split("val")
    ds_tr = C.ImageListDataset(train_samples, tf_train)
    ds_va = C.ImageListDataset(val_samples, tf_eval)

    g = torch.Generator()
    g.manual_seed(seed)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=C.BATCH_SIZE, shuffle=True,
                                        num_workers=0, generator=g, drop_last=False)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    return dl_tr, dl_va, val_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    C.freeze_all_bn(model)
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = probs.argmax(1)
    m = C.summary_metrics(labels, preds)
    return probs, labels, preds, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unfreeze", default="layer4", choices=["layer4", "layer3+4"])
    ap.add_argument("--preprocess", default="resize", choices=["resize", "pad"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS_DEFAULT)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run_name = f"{args.unfreeze.replace('+', '')}_{args.preprocess}_seed{args.seed}"
    if args.tag:
        run_name += f"_{args.tag}"
    run_dir = os.path.join(C.E4_ROOT, "runs", "e4_2", run_name)
    os.makedirs(run_dir, exist_ok=True)

    C.set_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C.load_e0(device)
    backbone_params, head_params = C.configure_trainable(model, args.unfreeze)
    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[{run_name}] device={device} trainable backbone={n_bb} head={n_hd}", flush=True)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": C.LR_BACKBONE})
    groups.append({"params": head_params, "lr": C.LR_HEAD})
    opt = torch.optim.Adam(groups)

    dl_tr, dl_va, val_samples = build_loaders(args.seed, args.preprocess)
    history = []
    best = {"epoch": -1, "val_macro_f1": -1.0}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        C.freeze_all_bn(model)
        tot_loss = 0.0
        tot_n = 0
        correct = 0
        t0 = time.time()
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * y.numel()
            tot_n += y.numel()
            correct += int((logits.argmax(1) == y).sum().item())
        train_loss = tot_loss / max(1, tot_n)
        train_acc = correct / max(1, tot_n)

        probs, labels, preds, m = evaluate(model, dl_va, device)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_weighted_f1": m["weighted_f1"],
            "val_trash_f1": m["per_class"]["trash"]["f1"],
            "n_correct": m["n_correct"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(f"[{run_name}] ep{epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={m['accuracy']:.4f} val_mF1={m['macro_f1']:.4f} "
              f"trashF1={rec['val_trash_f1']:.4f} ({rec['seconds']}s)", flush=True)

        if m["macro_f1"] > best["val_macro_f1"]:
            best = {"epoch": epoch, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]}

        if epoch == args.epochs:
            ckpt = os.path.join(run_dir, f"checkpoint_epoch{epoch:02d}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "run_name": run_name,
                "unfreeze": args.unfreeze,
                "preprocess": args.preprocess,
                "seed": args.seed,
                "val_macro_f1": m["macro_f1"],
                "val_accuracy": m["accuracy"],
            }, ckpt)
            np.savez(os.path.join(run_dir, "val_probs_epoch10.npz"),
                     probs=probs, labels=labels, preds=preds,
                     rel=np.array([s.rel for s in val_samples]))

    C.dump_json(os.path.join(run_dir, "EPOCH_HISTORY.json"), history)
    C.dump_json(os.path.join(run_dir, "RUN_CONFIG.json"), {
        "run_name": run_name,
        "unfreeze": args.unfreeze,
        "preprocess": args.preprocess,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": C.BATCH_SIZE,
        "lr_backbone": C.LR_BACKBONE,
        "lr_head": C.LR_HEAD,
        "optimizer": "Adam",
        "loss": "cross_entropy",
        "bn_mode": "eval, affine params and running stats frozen",
        "augmentation": "E0 train_transform (Resize224, HFlip, Rot15, ColorJitter)",
        "starting_checkpoint": C.E0_CKPT,
        "starting_checkpoint_sha256": C.sha256_file(C.E0_CKPT),
        "n_trainable_backbone": n_bb,
        "n_trainable_head": n_hd,
        "wall_seconds": round(time.time() - t_start, 1),
        "best_epoch_by_macro_f1": best,
        "final_epoch_metrics": history[-1] if history else None,
        "test_accessed": False,
    })
    print(f"[{run_name}] done in {time.time() - t_start:.1f}s -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
