"""Final Test evaluation (single, guarded access).

Runs exactly once, after every configuration has been frozen:
  * E0 reference on Test,
  * the final selected model / ensemble on Test,
  * paired McNemar between them,
  * writes TEST_ACCESSED.json + a row in logs/test_access.log.

Refuses to run twice unless --force is given (it must never be needed).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

OUT_DIR = os.path.join(C.E4_ROOT, "results", "final_test")
MARKER = os.path.join(C.E4_ROOT, "results", "final_test", "TEST_ACCESSED.json")
LOG = os.path.join(C.E4_ROOT, "logs", "test_access.log")


def hflip_tf(preprocess: str):
    base = C.eval_transform(preprocess == "resize")
    return transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)] + list(base.transforms))


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
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(MARKER) and not args.force:
        print(f"REFUSING: Test already accessed ({MARKER}).")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = C.list_split("test")
    tf = C.eval_transform(args.preprocess == "resize")
    dl = torch.utils.data.DataLoader(C.ImageListDataset(samples, tf), batch_size=C.BATCH_SIZE,
                                     shuffle=False, num_workers=0)
    dl_flip = None
    if args.tta == "hflip":
        dl_flip = torch.utils.data.DataLoader(
            C.ImageListDataset(samples, hflip_tf(args.preprocess)), batch_size=C.BATCH_SIZE,
            shuffle=False, num_workers=0)

    # --- E0 reference
    e0 = C.load_e0(device)
    e0.eval()
    C.freeze_all_bn(e0)
    p_e0, labels = predict(e0, dl, device)
    m_e0 = C.summary_metrics(labels, p_e0.argmax(1))
    print(f"E0 test: n={len(labels)} acc={m_e0['accuracy']:.4f} mF1={m_e0['macro_f1']:.4f}", flush=True)

    # --- final model(s)
    member_probs = []
    member_info = []
    for m in args.members:
        ck = torch.load(m, map_location="cpu", weights_only=False)
        model = C.make_resnet18_e0()
        model.load_state_dict(ck.get("model_state_dict", ck), strict=True)
        model.to(device).eval()
        C.freeze_all_bn(model)
        p, y = predict(model, dl, device)
        assert np.array_equal(labels, y)
        if dl_flip is not None:
            pf, _ = predict(model, dl_flip, device)
            p = 0.5 * (p + pf)
        member_probs.append(p)
        mm = C.summary_metrics(labels, p.argmax(1))
        member_info.append({"checkpoint": m, "sha256": C.sha256_file(m),
                            "test_accuracy": mm["accuracy"], "test_macro_f1": mm["macro_f1"]})
        print(f"member {os.path.basename(os.path.dirname(m))}: acc={mm['accuracy']:.4f} "
              f"mF1={mm['macro_f1']:.4f}", flush=True)

    final_prob = np.mean(member_probs, axis=0)
    m_fin = C.summary_metrics(labels, final_prob.argmax(1))
    mc = C.paired_mcnemar(p_e0.argmax(1), final_prob.argmax(1), labels)

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    result = {
        "timestamp": ts,
        "name": args.name,
        "preprocess": args.preprocess,
        "tta": args.tta,
        "n_members": len(args.members),
        "members": member_info,
        "n_test": int(len(labels)),
        "e0": {"accuracy": m_e0["accuracy"], "macro_f1": m_e0["macro_f1"],
               "weighted_f1": m_e0["weighted_f1"], "n_correct": m_e0["n_correct"],
               "per_class": m_e0["per_class"], "confusion_matrix": m_e0["confusion_matrix"]},
        "final": {"accuracy": m_fin["accuracy"], "macro_f1": m_fin["macro_f1"],
                  "weighted_f1": m_fin["weighted_f1"], "n_correct": m_fin["n_correct"],
                  "per_class": m_fin["per_class"], "confusion_matrix": m_fin["confusion_matrix"]},
        "delta": {
            "accuracy_pp": 100 * (m_fin["accuracy"] - m_e0["accuracy"]),
            "macro_f1_pp": 100 * (m_fin["macro_f1"] - m_e0["macro_f1"]),
            "weighted_f1_pp": 100 * (m_fin["weighted_f1"] - m_e0["weighted_f1"]),
            "n_correct": m_fin["n_correct"] - m_e0["n_correct"],
        },
        "mcnemar": mc,
        "class_order": list(C.CLASSES),
        "test_split": "Grouped Split V1 test (380), third_party/waste-classifier/data/test",
    }
    C.dump_json(os.path.join(OUT_DIR, f"{args.name}_test_results.json"), result)
    np.savez(os.path.join(OUT_DIR, f"{args.name}_test_probs.npz"),
             labels=labels, e0_probs=p_e0, final_probs=final_prob)
    C.dump_json(MARKER, {"timestamp": ts, "name": args.name, "members": args.members})
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts, "name": args.name, "n_test": int(len(labels)),
                            "members": len(args.members)}) + "\n")

    print(f"FINAL test: acc={m_fin['accuracy']:.4f} mF1={m_fin['macro_f1']:.4f} "
          f"(E0 {m_e0['accuracy']:.4f}/{m_e0['macro_f1']:.4f}) "
          f"delta_acc={result['delta']['accuracy_pp']:+.2f}pp p={mc['p_value']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
