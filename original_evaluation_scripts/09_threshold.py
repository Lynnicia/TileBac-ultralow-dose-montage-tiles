"""Phase 4 - honest operating point.

The published protocol fits the decision threshold on the TEST labels
(metrics_decision_tree_*.py `best_threshold`, called with the test arrays) and
reports F1 at that threshold. Here the same threshold is instead fitted on the
VALIDATION arrays and applied unchanged to test.

PX prob maps are built as the published scripts build them:
    prob_map[mask] = max(prob_map[mask], instance_score) / class
which is the construction in metrics_decision_tree_{d2,s3}.py. The same
construction is applied to every architecture so the two arms can compare.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

ROOT = "/home/vnw/tilebac_reanalysis"
OUT, PRED, WORK = f"{ROOT}/out", f"{ROOT}/out/predictions", f"{ROOT}/work"
MODELS = ["YOLOv11", "YOLO26", "U-Net", "Detectron2", "SAM3"]

def best_threshold(y_true, y_prob):
    """Published definition, unchanged (metrics_decision_tree_d2.py:328-331)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return thresholds[np.argmax(f1[:-1])]

def f1_at(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(np.uint8)
    tp = np.sum((pred == 1) & (y_true == 1)); fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))

def arrays(arch, res, split):
    """Return {class: (y_true, y_prob)} flattened pixel arrays."""
    gt_file = f"{WORK}/gt_{res}.json" if split == "test" else f"{WORK}/gt_{res}_{split}.json"
    coco = COCO(gt_file)
    preds = json.load(open(f"{PRED}/{arch}_{res}_{split}.json"))
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    T = {1: [], 2: []}; P = {1: [], 2: []}
    for iid in sorted(coco.getImgIds()):
        info = coco.imgs[iid]; H, W = info["height"], info["width"]
        gt = {1: np.zeros((H, W), np.uint8), 2: np.zeros((H, W), np.uint8)}
        for a in coco.loadAnns(coco.getAnnIds(imgIds=iid)):
            m = coco.annToMask(a)
            gt[a["category_id"]] |= m.astype(np.uint8)
        pm = {1: np.zeros((H, W), np.float32), 2: np.zeros((H, W), np.float32)}
        for p in by_img.get(iid, []):
            m = mask_utils.decode(p["segmentation"]).astype(bool)
            c = p["category_id"]
            pm[c][m] = np.maximum(pm[c][m], p["score"])
        for c in (1, 2):
            T[c].append(gt[c].ravel()); P[c].append(pm[c].ravel())
    return {c: (np.concatenate(T[c]), np.concatenate(P[c])) for c in (1, 2)}

CLS = {1: "IM", 2: "OM"}
rows = []
for res in ["640", "1024"]:
    for arch in MODELS:
        va = arrays(arch, res, "valid")
        te = arrays(arch, res, "test")
        # per class + combined "All" (concatenation of both classes, as published)
        items = [(CLS[c], va[c], te[c]) for c in (1, 2)]
        items.append(("All",
                      (np.concatenate([va[1][0], va[2][0]]), np.concatenate([va[1][1], va[2][1]])),
                      (np.concatenate([te[1][0], te[2][0]]), np.concatenate([te[1][1], te[2][1]]))))
        for cls, (vt, vp), (tt, tp) in items:
            thr_test = float(best_threshold(tt, tp))          # published: fitted on test
            f1_test = f1_at(tt, tp, thr_test)
            thr_val = float(best_threshold(vt, vp))           # honest: fitted on valid
            f1_val = f1_at(tt, tp, thr_val)                   # applied unchanged to test
            f1_half = f1_at(tt, tp, 0.5)                      # fixed reference threshold
            rows.append(dict(model=arch, res=int(res), **{"class": cls},
                             thr_fit_on_test=round(thr_test, 4),
                             f1_at_test_thr=round(f1_test, 4),
                             thr_fit_on_valid=round(thr_val, 4),
                             f1_at_valid_thr=round(f1_val, 4),
                             f1_at_fixed_0p5=round(f1_half, 4),
                             delta_f1=round(f1_test - f1_val, 4),
                             delta_f1_vs_fixed=round(f1_test - f1_half, 4)))
            print(f"{arch:11s} {res:>4s} {cls:>3s}  thr_test={thr_test:.3f} F1={f1_test:.4f} | "
                  f"thr_val={thr_val:.3f} F1={f1_val:.4f} | delta={f1_test-f1_val:+.4f}", flush=True)
        del va, te

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/07_threshold_comparison.csv", index=False)
print("\nwrote out/07_threshold_comparison.csv")
d = df.delta_f1
print(f"\nALL ROWS      n={len(df)}  mean delta_f1={d.mean():.4f}  max={d.max():.4f}  min={d.min():.4f}")
ex = df[~((df.model.isin(['YOLOv11','YOLO26'])) & (df['class'].isin(['OM','All'])))]
print(f"EXCL YOLO-OM  n={len(ex)}  mean delta_f1={ex.delta_f1.mean():.4f}  max={ex.delta_f1.max():.4f}")
print(f"\nrows moving > 0.02: {int((d > 0.02).sum())}")
print(df[df.delta_f1 > 0.02].to_string(index=False))
