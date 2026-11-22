"""Assemble the E4 deliverable tables from the per-experiment artefacts."""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

RES = os.path.join(C.E4_ROOT, "results")
FINAL = os.path.join(RES, "final_tables")
os.makedirs(FINAL, exist_ok=True)


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, f"({len(rows)} rows)")


def five_seed_table():
    runs = []
    for d in sorted(glob.glob(os.path.join(C.E4_ROOT, "runs", "e4_2", "layer4_resize_seed*"))):
        hp = os.path.join(d, "EPOCH_HISTORY.json")
        cp = os.path.join(d, "RUN_CONFIG.json")
        if not (os.path.exists(hp) and os.path.exists(cp)):
            continue
        cfg = load_json(cp)
        hist = load_json(hp)
        if len(hist) < 10:
            continue
        runs.append((os.path.basename(d), cfg, hist))
    rows = []
    per_class = []
    cms = []
    for name, cfg, hist in runs:
        last = hist[-1]
        z = np.load(os.path.join(C.E4_ROOT, "runs", "e4_2", name, "val_probs_epoch10.npz"), allow_pickle=True)
        m = C.summary_metrics(z["labels"], z["preds"])
        assert abs(m["accuracy"] - last["val_accuracy"]) < 1e-9
        rows.append({
            "run": name, "seed": cfg["seed"], "unfreeze": cfg["unfreeze"], "preprocess": cfg["preprocess"],
            "epochs": cfg["epochs"], "batch_size": cfg["batch_size"],
            "lr_backbone": cfg["lr_backbone"], "lr_head": cfg["lr_head"],
            "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_weighted_f1": m["weighted_f1"],
            "n_correct": m["n_correct"], "n_val": m["n"],
            "train_accuracy_ep10": last["train_accuracy"],
            "best_epoch_by_macro_f1": cfg["best_epoch_by_macro_f1"]["epoch"],
            "best_val_macro_f1": cfg["best_epoch_by_macro_f1"]["val_macro_f1"],
            "wall_seconds": cfg["wall_seconds"],
        })
        for cls, v in m["per_class"].items():
            per_class.append({"run": name, "seed": cfg["seed"], "class": cls,
                              "precision": v["precision"], "recall": v["recall"], "f1": v["f1"],
                              "support": v["support"]})
        cms.append(np.array(m["confusion_matrix"]))
    write_csv(os.path.join(FINAL, "E4_5SEED_PER_SEED.csv"), rows)

    agg = {}
    for k in ("val_accuracy", "val_macro_f1", "val_weighted_f1", "train_accuracy_ep10"):
        v = np.array([r[k] for r in rows])
        agg[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "n": int(len(v)),
                  "min": float(v.min()), "max": float(v.max())}
    pc = {}
    for cls in C.CLASSES:
        f1 = np.array([x["f1"] for x in per_class if x["class"] == cls])
        pr = np.array([x["precision"] for x in per_class if x["class"] == cls])
        rc = np.array([x["recall"] for x in per_class if x["class"] == cls])
        pc[cls] = {"f1_mean": float(f1.mean()), "f1_std": float(f1.std(ddof=1)),
                   "precision_mean": float(pr.mean()), "recall_mean": float(rc.mean())}
    pooled = np.sum(cms, axis=0)
    agg["per_class_mean"] = pc
    agg["pooled_confusion_matrix_5seeds"] = pooled.tolist()
    write_csv(os.path.join(FINAL, "E4_5SEED_PER_CLASS_MEAN.csv"),
              [{"class": c, **v} for c, v in pc.items()])
    write_csv(os.path.join(FINAL, "E4_5SEED_POOLED_CONFUSION.csv"),
              [{"true\\pred": C.CLASSES[i], **{C.CLASSES[j]: int(pooled[i, j]) for j in range(6)},
                "total": int(pooled[i].sum())} for i in range(6)])
    C.dump_json(os.path.join(FINAL, "E4_5SEED_SUMMARY.json"), agg)
    print("five-seed:", {k: (round(v['mean'], 4), round(v['std'], 4)) for k, v in agg.items() if k.startswith('val_') or k == 'train_accuracy_ep10'})
    return agg, rows


def e4_4_table():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "e4_4", "*.json"))):
        r = load_json(p)
        rows.append({
            "name": r["name"], "n_members": r["n_members"], "tta": r["tta"],
            "val_accuracy": r["ensemble"]["val_accuracy"],
            "val_macro_f1": r["ensemble"]["val_macro_f1"],
            "val_weighted_f1": r["ensemble"]["val_weighted_f1"],
            "n_correct": r["ensemble"]["n_correct"],
            "delta_vs_e0_correct": r["ensemble"]["delta_vs_e0_correct"],
            "mcnemar_p_vs_e0": r["ensemble"]["mcnemar_vs_e0"]["p_value"],
            "members": ";".join(os.path.basename(os.path.dirname(m["checkpoint"])) for m in r["members"]),
            "mean_pairwise_agreement": float(np.mean(r["pairwise_agreement"])) if r["pairwise_agreement"] else 1.0,
        })
    write_csv(os.path.join(FINAL, "E4_4_ENSEMBLE_TTA.csv"), rows)
    for r in rows:
        print(f"  {r['name']:26s} acc={r['val_accuracy']:.4f} mF1={r['val_macro_f1']:.4f} "
              f"correct={r['n_correct']} p={r['mcnemar_p_vs_e0']:.4f}")
    return rows


def e4_1_table():
    p = os.path.join(RES, "e4_1", "e4_1_results.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = []
    for r in rows:
        out.append({
            "feature_space": r["feature_space"], "classifier": r["classifier"],
            "best_params": r["best_params"],
            "cv_macro_f1": float(r["cv_macro_f1_mean"]),
            "val_accuracy": float(r["val_accuracy"]), "val_macro_f1": float(r["val_macro_f1"]),
            "val_weighted_f1": float(r["val_weighted_f1"]),
            "n_correct": int(r["n_correct"]), "delta_vs_e0": int(r["delta_vs_e0_correct"]),
            "mcnemar_p_vs_e0": float(r["mcnemar_p_vs_e0"]),
        })
    out.sort(key=lambda r: -r["val_macro_f1"])
    write_csv(os.path.join(FINAL, "E4_1_CLASSIFIERS.csv"), out)
    for r in out[:5]:
        print(f"  {r['feature_space']:5s} {r['classifier']:16s} mF1={r['val_macro_f1']:.4f} "
              f"acc={r['val_accuracy']:.4f} d={r['delta_vs_e0']:+d} p={r['mcnemar_p_vs_e0']:.4f}")
    return out


def e4_3_table():
    rows = []
    base = os.path.join(C.E4_ROOT, "runs", "e4_2")
    for seed in (42, 43):
        a = load_json(os.path.join(base, f"layer4_resize_seed{seed}", "EPOCH_HISTORY.json"))[-1]
        b = load_json(os.path.join(base, f"layer4_pad_seed{seed}", "EPOCH_HISTORY.json"))[-1]
        rows.append({
            "seed": seed,
            "resize_val_accuracy": a["val_accuracy"], "resize_val_macro_f1": a["val_macro_f1"],
            "pad_val_accuracy": b["val_accuracy"], "pad_val_macro_f1": b["val_macro_f1"],
            "delta_accuracy_pp": 100 * (b["val_accuracy"] - a["val_accuracy"]),
            "delta_macro_f1_pp": 100 * (b["val_macro_f1"] - a["val_macro_f1"]),
        })
    write_csv(os.path.join(FINAL, "E4_3_PREPROCESS.csv"), rows)
    for r in rows:
        print(f"  seed{r['seed']}: resize mF1={r['resize_val_macro_f1']:.4f} pad={r['pad_val_macro_f1']:.4f} "
              f"delta={r['delta_macro_f1_pp']:+.2f}pp")
    return rows


def e4_2_table():
    p = os.path.join(RES, "e4_2", "e4_2_runs.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    write_csv(os.path.join(FINAL, "E4_2_RUNS.csv"), rows)
    return rows


def main() -> int:
    print("== E4-1 ==");  e4_1_table()
    print("== E4-2 ==");  e4_2_table()
    print("== E4-3 ==");  e4_3_table()
    print("== E4-4 ==");  e4_4_table()
    print("== five seed =="); five_seed_table()
    e0 = load_json(os.path.join(RES, "e0_val_summary.json"))
    C.dump_json(os.path.join(FINAL, "E0_VAL_REFERENCE.json"), e0)
    print("E0 val:", e0["accuracy"], e0["n_correct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
