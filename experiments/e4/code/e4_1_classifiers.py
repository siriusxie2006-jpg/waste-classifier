"""E4-1: classifier optimisation on frozen E0 features.

Protocol
--------
* Features come from the frozen E0 backbone (256-D = ReLU output of fc[0];
  512-D supplementary = avgpool).
* Standardisation and the classifier are fit on Train only.
* Hyper-parameters are selected by 5-fold stratified CV *within Train*.
  Validation is used once per candidate for the final report, never for fitting.
* Every classifier's Validation predictions are stored for paired testing.

Test is never touched in this script.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as C

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEAT256_TRAIN = os.path.join(C.PROJECT_ROOT, "E2_FEATURE_DISTANCE_AUDIT", "FEATURES_TRAIN.csv")
FEAT256_VAL = os.path.join(C.PROJECT_ROOT, "E2_FEATURE_DISTANCE_AUDIT", "FEATURES_VALIDATION.csv")
FEAT512_TRAIN = os.path.join(C.PROJECT_ROOT, "E2_FORMAL_PHASE1", "A6_FEATURES_TRAIN.csv")
FEAT512_VAL = os.path.join(C.PROJECT_ROOT, "E2_FORMAL_PHASE1", "A6_FEATURES_VALIDATION.csv")

OUT_DIR = os.path.join(C.E4_ROOT, "results", "e4_1")
os.makedirs(OUT_DIR, exist_ok=True)


def load_features(path: str, dim: int):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    rel_col = header.index("relative_path")
    cls_col = header.index("class")
    feat_cols = [i for i, h in enumerate(header) if h.startswith("feature_")]
    assert len(feat_cols) == dim, (path, len(feat_cols))
    X = np.array([[float(row[i]) for i in feat_cols] for row in rows], dtype=np.float64)
    y = np.array([C.CLASSES.index(row[cls_col]) for row in rows], dtype=np.int64)
    rel = []
    for row in rows:
        r = row[rel_col].replace("\\", "/")
        parts = r.split("/")
        if parts and parts[0] in ("train", "val", "test"):
            r = "/".join(parts[1:])
        rel.append(r)
    return X, y, rel


def align_to(reference_rel, X, y, rel):
    """Reorder feature rows to match reference_rel order."""
    idx = {r: i for i, r in enumerate(rel)}
    order = np.array([idx[r] for r in reference_rel], dtype=np.int64)
    return X[order], y[order]


def make_candidates():
    """name -> list of (params_dict, estimator_factory)."""
    cands: dict[str, list[tuple[dict, object]]] = {}
    cands["logreg"] = [
        ({"C": c}, Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=5000))]))
        for c in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]
    cands["linear_svm"] = [
        ({"C": c}, Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="linear", C=c))]))
        for c in (0.001, 0.01, 0.1, 1.0, 10.0)
    ]
    cands["rbf_svm"] = [
        ({"C": c, "gamma": g},
         Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=c, gamma=g))]))
        for c, g in itertools.product((1.0, 10.0, 100.0), ("scale", 0.001, 0.01))
    ]
    cands["knn_euclid"] = [
        ({"k": k}, Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))]))
        for k in (1, 3, 5, 9, 15)
    ]
    cands["knn_cosine"] = [
        ({"k": k}, Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k, metric="cosine"))]))
        for k in (1, 3, 5, 9, 15)
    ]
    cands["lda_shrinkage"] = [
        ({"shrinkage": s},
         Pipeline([("sc", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=s))]))
        for s in ("auto", 0.1, 0.3, 0.5, 0.8)
    ]
    cands["nearest_centroid"] = [
        ({"metric": "euclidean"}, Pipeline([("sc", StandardScaler()), ("clf", NearestCentroid())]))
    ]
    return cands


def cv_select(Xtr, ytr, cand_list, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(Xtr, ytr))
    best = None
    table = []
    for params, est in cand_list:
        f1s, accs = [], []
        for tr_idx, te_idx in splits:
            est.fit(Xtr[tr_idx], ytr[tr_idx])
            pred = est.predict(Xtr[te_idx])
            f1s.append(f1_score(ytr[te_idx], pred, average="macro", zero_division=0))
            accs.append(accuracy_score(ytr[te_idx], pred))
        row = dict(params)
        row.update({"cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s)),
                    "cv_accuracy_mean": float(np.mean(accs))})
        table.append(row)
        key = (row["cv_macro_f1_mean"], row["cv_accuracy_mean"])
        if best is None or key > best[0]:
            best = (key, params, est)
    return best[1], best[2], table


def run_feature_space(tag: str, ftr: str, fva: str, dim: int, ref_rel, ref_labels, e0_preds):
    Xtr, ytr, rel_tr = load_features(ftr, dim)
    Xva, yva, rel_va = load_features(fva, dim)
    Xtr, ytr = align_to(ref_rel["train"], Xtr, ytr, rel_tr)
    Xva, yva = align_to(ref_rel["val"], Xva, yva, rel_va)
    assert np.array_equal(yva, ref_labels), "label order mismatch"

    cands = make_candidates()
    results = []
    all_preds = {}
    for fam, cand_list in cands.items():
        t0 = __import__("time").time()
        best_params, _est, table = cv_select(Xtr, ytr, cand_list)
        # refit the winning configuration on the full Train split
        chosen = None
        for params, est in cand_list:
            if params == best_params:
                chosen = est
                break
        assert chosen is not None
        chosen.fit(Xtr, ytr)
        pred = chosen.predict(Xva)
        proba = None
        if hasattr(chosen, "predict_proba"):
            try:
                proba = chosen.predict_proba(Xva)
            except Exception:
                proba = None
        m = C.summary_metrics(yva, pred)
        mc = C.paired_mcnemar(e0_preds, pred, yva)
        n_params = len(cand_list)
        results.append({
            "feature_space": tag,
            "classifier": fam,
            "best_params": best_params,
            "cv_macro_f1_mean": max(x["cv_macro_f1_mean"] for x in table),
            "cv_accuracy_mean": max(x["cv_accuracy_mean"] for x in table),
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_weighted_f1": m["weighted_f1"],
            "n_correct": m["n_correct"],
            "delta_vs_e0_correct": m["n_correct"] - int((e0_preds == yva).sum()),
            "mcnemar_p_vs_e0": mc["p_value"],
            "mcnemar_b": mc["b"],
            "mcnemar_c": mc["c"],
            "seconds": round(__import__("time").time() - t0, 1),
            "n_configs": n_params,
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
            "cv_table": table,
        })
        all_preds[f"{tag}::{fam}"] = pred
        print(f"[{tag}] {fam:16s} cv_mF1={results[-1]['cv_macro_f1_mean']:.4f} "
              f"val_acc={m['accuracy']:.4f} val_mF1={m['macro_f1']:.4f} "
              f"n_correct={m['n_correct']} p={mc['p_value']:.4f} ({results[-1]['seconds']}s)")

    return results, all_preds, yva


def main() -> int:
    # reference ordering + E0 predictions
    val_samples = C.list_split("val")
    train_samples = C.list_split("train")
    ref_rel = {"val": [s.rel for s in val_samples], "train": [s.rel for s in train_samples]}

    z = np.load(os.path.join(C.E4_ROOT, "results", "e0_val_probs.npz"), allow_pickle=True)
    assert list(z["rel"]) == ref_rel["val"], "E0 val order mismatch"
    e0_preds = z["preds"]
    yva = z["labels"]
    print(f"E0 reference: n={len(yva)} correct={(e0_preds == yva).sum()} acc={(e0_preds == yva).mean():.4f}")

    all_results = []
    preds_store = {"__labels__": yva, "__e0__": e0_preds}

    r256, p256, _ = run_feature_space("256D", FEAT256_TRAIN, FEAT256_VAL, 256, ref_rel, yva, e0_preds)
    all_results += r256
    preds_store.update(p256)

    r512, p512, _ = run_feature_space("512D", FEAT512_TRAIN, FEAT512_VAL, 512, ref_rel, yva, e0_preds)
    all_results += r512
    preds_store.update(p512)

    np.savez(os.path.join(OUT_DIR, "e4_1_val_predictions.npz"),
             labels=yva, e0=e0_preds,
             **{k: v for k, v in preds_store.items() if not k.startswith("__")})

    # summary CSV (flat, per-class metrics separate)
    fields = ["feature_space", "classifier", "best_params", "cv_macro_f1_mean", "cv_accuracy_mean",
              "val_accuracy", "val_macro_f1", "val_weighted_f1", "n_correct",
              "delta_vs_e0_correct", "mcnemar_b", "mcnemar_c", "mcnemar_p_vs_e0", "seconds", "n_configs"]
    with open(os.path.join(OUT_DIR, "e4_1_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], dict) else r[k]) for k in fields})

    C.dump_json(os.path.join(OUT_DIR, "e4_1_results.json"), all_results)

    # rank and pick
    ranked = sorted(all_results, key=lambda r: (r["val_macro_f1"], r["val_accuracy"]), reverse=True)
    print("\n=== ranked by Validation macro-F1 ===")
    for r in ranked[:6]:
        print(f"  {r['feature_space']:5s} {r['classifier']:16s} mF1={r['val_macro_f1']:.4f} "
              f"acc={r['val_accuracy']:.4f} d_correct={r['delta_vs_e0_correct']:+d} p={r['mcnemar_p_vs_e0']:.4f}")
    C.dump_json(os.path.join(OUT_DIR, "e4_1_selected.json"), {
        "rule": "highest Validation macro-F1, tie-break Validation accuracy",
        "selected": ranked[0],
        "e0": {"val_accuracy": float((e0_preds == yva).mean()),
               "n_correct": int((e0_preds == yva).sum())},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
