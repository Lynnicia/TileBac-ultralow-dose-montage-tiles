"""
Geometry reference baselines and model Dice/IoU.

Adapted from the null-baseline section of the original
17_null_and_geometry.py.

Reference masks:
    filled bounding box
    ellipse
    convex hull

Each model prediction is compared with the ground-truth
instance of the same class using the best Dice match.
Predictions are filtered at score >= 0.5.
"""

import json
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from config import (
    WORK_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
)


MODELS = [
    "YOLOv11",
    "YOLO26",
    "U-Net",
    "Detectron2",
    "SAM3",
]

CLS = {
    1: "IM",
    2: "OM",
}


def dice(a, b):
    s = a.sum() + b.sum()

    return (
        float(2 * (a & b).sum() / s)
        if s
        else 0.0
    )


def iou(a, b):
    u = (a | b).sum()

    return (
        float((a & b).sum() / u)
        if u
        else 0.0
    )


def bbox_mask(g, H, W):

    ys, xs = np.nonzero(g)

    m = np.zeros(
        (H, W),
        bool,
    )

    m[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1,
    ] = True

    return m


def ellipse_mask(g, H, W):

    ys, xs = np.nonzero(g)

    m = np.zeros(
        (H, W),
        np.uint8,
    )

    cv2.ellipse(
        m,
        (
            (
                int(
                    (xs.min() + xs.max()) / 2
                ),
                int(
                    (ys.min() + ys.max()) / 2
                ),
            ),
            (
                int(
                    xs.max() - xs.min() + 1
                ),
                int(
                    ys.max() - ys.min() + 1
                ),
            ),
            0,
        ),
        1,
        -1,
    )

    return m.astype(bool)


def hull_mask(g, H, W):

    cnts, _ = cv2.findContours(
        g.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    pts = np.vstack(cnts)

    h = cv2.convexHull(pts)

    m = np.zeros(
        (H, W),
        np.uint8,
    )

    cv2.fillPoly(
        m,
        [h],
        1,
    )

    return m.astype(bool)


# ============================================================
# NULL / GEOMETRY BASELINES
# ============================================================

rows = []

for res in [
    "640",
    "1024",
]:

    gt = COCO(
        str(
            WORK_DIR
            / f"gt_{res}.json"
        )
    )

    Pm = {}

    for m in MODELS:

        d = {}

        pred_file = (
            PREDICTION_DIR
            / f"{m}_{res}_test.json"
        )

        with pred_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            preds = json.load(f)

        for x in preds:

            d.setdefault(
                x["image_id"],
                [],
            ).append(x)

        Pm[m] = d

    for iid in sorted(
        gt.getImgIds()
    ):

        info = gt.imgs[iid]

        H = info["height"]
        W = info["width"]

        # Preserve original score >= 0.5 filtering.
        dec = {
            m: [
                (
                    x["category_id"],
                    mask_utils.decode(
                        x["segmentation"]
                    ).astype(bool),
                    float(x["score"]),
                )
                for x in Pm[m].get(
                    iid,
                    [],
                )
                if x["score"] >= 0.5
            ]
            for m in MODELS
        }

        anns = gt.loadAnns(
            gt.getAnnIds(
                imgIds=iid
            )
        )

        for k, a in enumerate(
            anns,
            1,
        ):

            g = gt.annToMask(
                a
            ).astype(bool)

            if not g.any():
                continue

            cid = a["category_id"]
            cn = CLS[cid]

            base = dict(
                res=int(res),
                image_id=iid,
                instance=k,
                cls=cn,
                gt_area=int(
                    g.sum()
                ),
            )

            # --------------------------------------------
            # Geometry reference predictors
            # --------------------------------------------

            for name, mm in [

                (
                    "null_bbox_fill",
                    bbox_mask(
                        g,
                        H,
                        W,
                    ),
                ),

                (
                    "null_ellipse",
                    ellipse_mask(
                        g,
                        H,
                        W,
                    ),
                ),

                (
                    "null_convex_hull",
                    hull_mask(
                        g,
                        H,
                        W,
                    ),
                ),
            ]:

                rows.append(
                    dict(
                        **base,
                        method=name,
                        dice=round(
                            dice(mm, g),
                            5,
                        ),
                        iou=round(
                            iou(mm, g),
                            5,
                        ),
                        kind="null",
                    )
                )

            # --------------------------------------------
            # Model predictions
            # --------------------------------------------

            for m in MODELS:

                cand = [
                    p
                    for c, p, s
                    in dec[m]
                    if c == cid
                ]

                if cand:

                    ds = [
                        dice(p, g)
                        for p in cand
                    ]

                    j = int(
                        np.argmax(ds)
                    )

                    rows.append(
                        dict(
                            **base,
                            method=m,
                            dice=round(
                                ds[j],
                                5,
                            ),
                            iou=round(
                                iou(
                                    cand[j],
                                    g,
                                ),
                                5,
                            ),
                            kind="model",
                        )
                    )

                else:

                    rows.append(
                        dict(
                            **base,
                            method=m,
                            dice=0.0,
                            iou=0.0,
                            kind="model",
                        )
                    )


# ============================================================
# PER-INSTANCE RESULTS
# ============================================================

d = pd.DataFrame(rows)

per_instance_file = (
    OUTPUT_DIR
    / "05_geometry_per_instance.csv"
)

d.to_csv(
    per_instance_file,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

def ci95(x):

    x = np.asarray(
        x,
        float,
    )

    n = len(x)

    if n < 2:
        return (
            np.nan,
            np.nan,
        )

    se = (
        x.std(ddof=1)
        / np.sqrt(n)
    )

    return (
        x.mean() - 1.96 * se,
        x.mean() + 1.96 * se,
    )


srows = []


# Per-class summaries
for (
    res,
    cls,
    meth,
), g in d.groupby(
    [
        "res",
        "cls",
        "method",
    ]
):

    lo, hi = ci95(
        g.dice
    )

    srows.append(
        dict(
            res=res,
            cls=cls,
            method=meth,
            kind=g.kind.iloc[0],
            n=len(g),

            mean_dice=round(
                g.dice.mean(),
                5,
            ),

            ci95_lo=round(
                lo,
                5,
            ),

            ci95_hi=round(
                hi,
                5,
            ),

            median_dice=round(
                g.dice.median(),
                5,
            ),

            mean_iou=round(
                g.iou.mean(),
                5,
            ),
        )
    )


# Combined IM + OM summaries
for (
    res,
    meth,
), g in d.groupby(
    [
        "res",
        "method",
    ]
):

    lo, hi = ci95(
        g.dice
    )

    srows.append(
        dict(
            res=res,
            cls="All",
            method=meth,
            kind=g.kind.iloc[0],
            n=len(g),

            mean_dice=round(
                g.dice.mean(),
                5,
            ),

            ci95_lo=round(
                lo,
                5,
            ),

            ci95_hi=round(
                hi,
                5,
            ),

            median_dice=round(
                g.dice.median(),
                5,
            ),

            mean_iou=round(
                g.iou.mean(),
                5,
            ),
        )
    )


s = pd.DataFrame(
    srows
).sort_values(
    [
        "res",
        "cls",
        "mean_dice",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)


summary_file = (
    OUTPUT_DIR
    / "05_geometry_summary.csv"
)

s.to_csv(
    summary_file,
    index=False,
)


print(
    "=== geometry reference summary "
    "(All classes) ==="
)

print(
    s[
        s.cls == "All"
    ][
        [
            "res",
            "method",
            "kind",
            "n",
            "mean_dice",
            "ci95_lo",
            "ci95_hi",
        ]
    ].to_string(
        index=False
    )
)

print(
    f"\nwrote {per_instance_file}"
)

print(
    f"wrote {summary_file}"
)