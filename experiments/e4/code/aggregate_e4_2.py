"""Aggregate E4-2/E4-3 run results into CSV + JSON summaries (no training)."""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

RUNS = os.path.join(C.E4_ROOT, "runs", "e4_2")
OUT = os.path.join(C.E4_ROOT, "results", "e4_2")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        cfg_p = os.path.join(d, "RUN_CONFIG.json")
        hist_p = os.path.join(d, "EPOCH_HISTORY.json")
        if not (os.path.exists(cfg_p) and os.path.exists(hist_p)):
            continue
        cfg = json.load(open(cfg_p, encoding="utf-8"))
        hist = json.load(open(hist_p, encoding="utf-8"))
        if len(hist) < cfg.get("epochs", 10) or "_smoke" in d:
            continue
        last = hist[-1]
        best = max(hist, key=lambda r: r["val_macro_f1"])
        rows.append({
            "run": os.path.basename(d),
            "unfreeze": cfg["unfreeze"],
            "preprocess": cfg["preprocess"],
            "seed": cfg["seed"],
            "val_accuracy_ep10": last["val_accuracy"],
            "val_macro_f1_ep10": last["val_macro_f1"],
            "val_weighted_f1_ep10": last["val_weighted_f1"],
            "val_trash_f1_ep10": last["val_trash_f1"],
            "n_correct_ep10": last["n_correct"],
            "train_accuracy_ep10": last["train_accuracy"],
            "train_loss_ep10": last["train_loss"],
            "best_epoch": best["epoch"],
            "best_val_macro_f1": best["val_macro_f1"],
            "seconds": cfg.get("wall_seconds"),
        })

    if not rows:
        print("no completed runs yet")
        return 0

    fields = list(rows[0].keys())
    with open(os.path.join(OUT, "e4_2_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # per (unfreeze, preprocess) summary
    summary = {}
    for r in rows:
        key = f"{r['unfreeze']}|{r['preprocess']}"
        summary.setdefault(key, []).append(r)
    out = {}
    for key, rs in summary.items():
        def stats(field):
            v = np.array([x[field] for x in rs], dtype=float)
            return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
        out[key] = {
            "n_seeds": len(rs),
            "seeds": [x["seed"] for x in rs],
            "val_accuracy": stats("val_accuracy_ep10"),
            "val_macro_f1": stats("val_macro_f1_ep10"),
            "val_weighted_f1": stats("val_weighted_f1_ep10"),
            "val_trash_f1": stats("val_trash_f1_ep10"),
            "train_accuracy": stats("train_accuracy_ep10"),
            "per_seed_macro_f1": {x["seed"]: x["val_macro_f1_ep10"] for x in rs},
            "per_seed_trash_f1": {x["seed"]: x["val_trash_f1_ep10"] for x in rs},
        }
    C.dump_json(os.path.join(OUT, "e4_2_summary.json"), out)

    print(f"{'run':40s} {'acc':>7s} {'mF1':>7s} {'trashF1':>8s} {'trAcc':>7s} {'bestEp':>6s}")
    for r in rows:
        print(f"{r['run']:40s} {r['val_accuracy_ep10']:.4f} {r['val_macro_f1_ep10']:.4f} "
              f"{r['val_trash_f1_ep10']:.4f} {r['train_accuracy_ep10']:.4f} {r['best_epoch']:6d}")
    print("\n--- summary by config ---")
    for k, v in sorted(out.items()):
        print(f"{k:20s} n={v['n_seeds']} acc={v['val_accuracy']['mean']:.4f}+-{v['val_accuracy']['std']:.4f} "
              f"mF1={v['val_macro_f1']['mean']:.4f}+-{v['val_macro_f1']['std']:.4f} "
              f"trash={v['val_trash_f1']['mean']:.4f}+-{v['val_trash_f1']['std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
