"""Phase 10.1 - fill-and-score diagnostic.

Applies morphological closing + hole filling to all predicted mask of each 10
models, then rescores with the ID COCOeval call used in Phase 3.

Note here, if YOLO's OM output is a closed boundary trace we can fill it to recover a
region mask and AP50 should jump. If the rings are broken, filling does nothing.
Models that already emit filled regions should be near-unchanged (control).
"""

import contextlib
import io
import json

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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

CLS = {1: "IM", 2: "OM"}

R_MAIN = 5
R_SWEEP = [3, 5, 9]


def kern(r):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * r + 1, 2 * r + 1),
    )


def fill(m, r):
    c = cv2.morphologyEx(
        m.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kern(r),
    )

    return ndimage.binary_fill_holes(
        c.astype(bool)
    )


def enc(m):
    r = mask_utils.encode(
        np.asfortranarray(
            m.astype(np.uint8)
        )
    )

    r["counts"] = r["counts"].decode("ascii")

    return r


def ap50(gt_json, preds, cat=None):

    # loadRes mutates the ann dicts in place (adds numpy 'bbox'),
    # which breaks a second call on the same list.
    # Therefore pass a shallow copy of each dict.

    if not preds:
        return np.nan

    preds = [dict(x) for x in preds]

    gt = COCO(str(gt_json))

    with contextlib.redirect_stdout(io.StringIO()):

        dt = gt.loadRes(preds)
        e = COCOeval(gt, dt, "segm")

        if cat is not None:
            e.params.catIds = [cat]

        e.evaluate()
        e.accumulate()
        e.summarize()

    return float(e.stats[1])


rows = []
ringrows = []


for res in ["640", "1024"]:

    gtj = WORK_DIR / f"gt_{res}.json"

    for m in MODELS:

        pred_file = (
            PREDICTION_DIR
            / f"{m}_{res}_test.json"
        )

        with pred_file.open("r", encoding="utf-8") as f:
            src = json.load(f)

        filled = []
        stats = []

        for x in src:

            b = mask_utils.decode(
                x["segmentation"]
            ).astype(bool)

            f = fill(b, R_MAIN)

            filled.append(
                dict(
                    image_id=x["image_id"],
                    category_id=x["category_id"],
                    segmentation=enc(f),
                    score=x["score"],
                )
            )

            a0 = int(b.sum())
            a1 = int(f.sum())

            ncc0 = (
                cv2.connectedComponents(
                    b.astype(np.uint8)
                )[0]
                - 1
            )

            ncc1 = (
                cv2.connectedComponents(
                    f.astype(np.uint8)
                )[0]
                - 1
            )

            stats.append(
                (
                    x["category_id"],
                    x["score"],
                    a0,
                    a1,
                    ncc0,
                    ncc1,
                )
            )


        filled_file = (
            PREDICTION_DIR
            / f"{m}_{res}_test_FILLED.json"
        )

        with filled_file.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(filled, f)


        st = pd.DataFrame(
            stats,
            columns=[
                "cat",
                "score",
                "a_before",
                "a_after",
                "ncc_before",
                "ncc_after",
            ],
        )


        for cid, cn in [
            (None, "All"),
            (1, "IM"),
            (2, "OM"),
        ]:

            b = ap50(gtj, src, cid)
            a = ap50(gtj, filled, cid)

            sub = (
                st
                if cid is None
                else st[st.cat == cid]
            )

            hi = sub[sub.score >= 0.5]

            rows.append(
                dict(
                    model=m,
                    res=int(res),
                    cls=cn,
                    ap50_before=round(b, 4),
                    ap50_after=round(a, 4),
                    delta=round(a - b, 4),
                    n_pred=len(sub),
                    median_area_before=(
                        int(sub.a_before.median())
                        if len(sub)
                        else 0
                    ),
                    median_area_after=(
                        int(sub.a_after.median())
                        if len(sub)
                        else 0
                    ),
                    area_gain_ratio=round(
                        float(
                            sub.a_after.sum()
                            / max(
                                sub.a_before.sum(),
                                1,
                            )
                        ),
                        3,
                    ),
                    n_conf50=len(hi),
                )
            )


        print(
            f"{m:11s} {res:>4s} "
            f"All: {rows[-3]['ap50_before']:.3f} -> "
            f"{rows[-3]['ap50_after']:.3f} "
            f"({rows[-3]['delta']:+.3f})  |  "
            f"OM: {rows[-1]['ap50_before']:.3f} -> "
            f"{rows[-1]['ap50_after']:.3f} "
            f"({rows[-1]['delta']:+.3f})",
            flush=True,
        )


        # ring closure stats, confident predictions only

        for r in R_SWEEP:

            for cid, cn in [
                (1, "IM"),
                (2, "OM"),
            ]:

                sel = [
                    x
                    for x in src
                    if x["category_id"] == cid
                    and x["score"] >= 0.5
                ]

                closed = 0
                open_ = 0

                for x in sel:

                    b = mask_utils.decode(
                        x["segmentation"]
                    ).astype(bool)

                    f = fill(b, r)

                    # "closed" = filling produced a real interior
                    # (>=2x the traced area)

                    if f.sum() >= 2 * max(
                        b.sum(),
                        1,
                    ):
                        closed += 1
                    else:
                        open_ += 1

                n = closed + open_

                ringrows.append(
                    dict(
                        model=m,
                        res=int(res),
                        cls=cn,
                        closing_radius_px=r,
                        n_conf50=n,
                        n_closed=closed,
                        n_open=open_,
                        frac_closed=(
                            round(closed / n, 4)
                            if n
                            else np.nan
                        ),
                    )
                )


df = pd.DataFrame(rows)
rg = pd.DataFrame(ringrows)


df.to_csv(
    OUTPUT_DIR / "10_fill_and_score.csv",
    index=False,
)

rg.to_csv(
    OUTPUT_DIR / "10_ring_closure.csv",
    index=False,
)


print(
    "\n=== YOLO OM ring closure "
    "(score>=0.5) ==="
)

print(
    rg[
        (rg.model.isin(["YOLOv11", "YOLO26"]))
        & (rg.cls == "OM")
    ][
        [
            "model",
            "res",
            "closing_radius_px",
            "n_conf50",
            "n_closed",
            "n_open",
            "frac_closed",
        ]
    ].to_string(index=False)
)

print(
    "\nwrote 10_fill_and_score.csv "
    "and 10_ring_closure.csv"
)