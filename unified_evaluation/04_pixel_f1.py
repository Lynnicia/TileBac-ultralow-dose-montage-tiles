"""
Compare F1 scores using three threshold-selection strategies.

For each model, pixel-level probability maps are reconstructed from the
predicted instance masks and confidence scores. F1 is then evaluated on the
test set using:

1. A threshold selected from the test set.
2. A threshold selected from the validation set and then applied to the test set.
3. A fixed threshold of 0.5.

This allows us to measure how much the reported F1 changes depending on how the decision threshold is selected. 
The same procedure is applied to every model for a consistent comparison.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

from config import WORK_DIR, OUTPUT_DIR, PREDICTION_DIR


MODELS = [
    "YOLOv11",
    "YOLO26",
    "U-Net",
    "Detectron2",
    "SAM3",
]

def best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_prob,
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-8)
    )

    return thresholds[np.argmax(f1[:-1])]


def f1_at(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(np.uint8)

    tp = np.sum(
        (pred == 1) & (y_true == 1)
    )
    fp = np.sum(
        (pred == 1) & (y_true == 0)
    )
    fn = np.sum(
        (pred == 0) & (y_true == 1)
    )

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)

    return float(
        2 * p * r / (p + r + 1e-8)
    )


def arrays(arch, res, split):
    """
    Return:
        {class: (y_true, y_prob)}

    as flattened pixel arrays.
    """

    # Preserve the original ground-truth naming:
    # test  -> gt_640.json
    # valid -> gt_640_valid.json
    if split == "test":
        gt_file = WORK_DIR / f"gt_{res}.json"
    else:
        gt_file = WORK_DIR / f"gt_{res}_{split}.json"

    pred_file = (
        PREDICTION_DIR
        / f"{arch}_{res}_{split}.json"
    )

    if not gt_file.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n{gt_file}"
        )

    if not pred_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{pred_file}"
        )

    coco = COCO(str(gt_file))

    with pred_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        preds = json.load(f)

    by_img = {}

    for p in preds:
        by_img.setdefault(
            p["image_id"],
            [],
        ).append(p)

    T = {
        1: [],
        2: [],
    }

    P = {
        1: [],
        2: [],
    }

    for iid in sorted(coco.getImgIds()):

        info = coco.imgs[iid]

        H = info["height"]
        W = info["width"]

        gt = {
            1: np.zeros(
                (H, W),
                np.uint8,
            ),
            2: np.zeros(
                (H, W),
                np.uint8,
            ),
        }

        for a in coco.loadAnns(
            coco.getAnnIds(imgIds=iid)
        ):
            m = coco.annToMask(a)

            gt[a["category_id"]] |= (
                m.astype(np.uint8)
            )

        pm = {
            1: np.zeros(
                (H, W),
                np.float32,
            ),
            2: np.zeros(
                (H, W),
                np.float32,
            ),
        }

        for p in by_img.get(iid, []):

            m = mask_utils.decode(
                p["segmentation"]
            ).astype(bool)

            c = p["category_id"]

            pm[c][m] = np.maximum(
                pm[c][m],
                p["score"],
            )

        for c in (1, 2):
            T[c].append(
                gt[c].ravel()
            )

            P[c].append(
                pm[c].ravel()
            )

    return {
        c: (
            np.concatenate(T[c]),
            np.concatenate(P[c]),
        )
        for c in (1, 2)
    }


CLS = {
    1: "IM",
    2: "OM",
}

rows = []

for res in ["640", "1024"]:

    for arch in MODELS:

        va = arrays(
            arch,
            res,
            "valid",
        )

        te = arrays(
            arch,
            res,
            "test",
        )

        # Per class + combined "All"
        # (concatenation of both classes, as published)
        items = [
            (
                CLS[c],
                va[c],
                te[c],
            )
            for c in (1, 2)
        ]

        items.append(
            (
                "All",
                (
                    np.concatenate(
                        [
                            va[1][0],
                            va[2][0],
                        ]
                    ),
                    np.concatenate(
                        [
                            va[1][1],
                            va[2][1],
                        ]
                    ),
                ),
                (
                    np.concatenate(
                        [
                            te[1][0],
                            te[2][0],
                        ]
                    ),
                    np.concatenate(
                        [
                            te[1][1],
                            te[2][1],
                        ]
                    ),
                ),
            )
        )

        for cls, (vt, vp), (tt, tp) in items:

            # Published:
            # threshold fitted on test
            thr_test = float(
                best_threshold(
                    tt,
                    tp,
                )
            )

            f1_test = f1_at(
                tt,
                tp,
                thr_test,
            )

            # Validation-fitted threshold
            thr_val = float(
                best_threshold(
                    vt,
                    vp,
                )
            )

            # Apply unchanged to test
            f1_val = f1_at(
                tt,
                tp,
                thr_val,
            )

            # Fixed reference threshold
            f1_half = f1_at(
                tt,
                tp,
                0.5,
            )

            rows.append(
                dict(
                    model=arch,
                    res=int(res),
                    **{"class": cls},
                    thr_fit_on_test=round(
                        thr_test,
                        4,
                    ),
                    f1_at_test_thr=round(
                        f1_test,
                        4,
                    ),
                    thr_fit_on_valid=round(
                        thr_val,
                        4,
                    ),
                    f1_at_valid_thr=round(
                        f1_val,
                        4,
                    ),
                    f1_at_fixed_0p5=round(
                        f1_half,
                        4,
                    ),
                    delta_f1=round(
                        f1_test - f1_val,
                        4,
                    ),
                    delta_f1_vs_fixed=round(
                        f1_test - f1_half,
                        4,
                    ),
                )
            )

            print(
                f"{arch:11s} "
                f"{res:>4s} "
                f"{cls:>3s}  "
                f"thr_test={thr_test:.3f} "
                f"F1={f1_test:.4f} | "
                f"thr_val={thr_val:.3f} "
                f"F1={f1_val:.4f} | "
                f"delta={f1_test-f1_val:+.4f}",
                flush=True,
            )

        del va, te


df = pd.DataFrame(rows)

output_file = (
    OUTPUT_DIR
    / "07_threshold_comparison.csv"
)

df.to_csv(
    output_file,
    index=False,
)

print(
    f"\nwrote {output_file}"
)


d = df.delta_f1

print(
    f"\nALL ROWS      "
    f"n={len(df)}  "
    f"mean delta_f1={d.mean():.4f}  "
    f"max={d.max():.4f}  "
    f"min={d.min():.4f}"
)


ex = df[
    ~(
        (
            df.model.isin(
                [
                    "YOLOv11",
                    "YOLO26",
                ]
            )
        )
        & (
            df["class"].isin(
                [
                    "OM",
                    "All",
                ]
            )
        )
    )
]

print(
    f"EXCL YOLO-OM  "
    f"n={len(ex)}  "
    f"mean delta_f1="
    f"{ex.delta_f1.mean():.4f}  "
    f"max={ex.delta_f1.max():.4f}"
)

print(
    f"\nrows moving > 0.02: "
    f"{int((d > 0.02).sum())}"
)

print(
    df[
        df.delta_f1 > 0.02
    ].to_string(
        index=False
    )
)