"""
Unified COCO evaluation for all TileBac models.

One COCOeval pass with identical parameters for every model.

AP50 = coco_eval.stats[1]
(never hand-indexed from eval["precision"])
"""

import contextlib
import io
import json

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import (
    WORK_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
)


CATS = {
    1: "IM",
    2: "OM",
}

MODELS = [
    "YOLOv11",
    "YOLO26",
    "U-Net",
    "Detectron2",
    "SAM3",
]


def evaluate(gt_json, dt_json, cat_ids=None):
    """
    Run COCO segmentation evaluation.

    cat_ids=None evaluates all classes.
    cat_ids=[1] evaluates IM.
    cat_ids=[2] evaluates OM.
    """

    coco_gt = COCO(str(gt_json))

    with dt_json.open("r", encoding="utf-8") as f:
        preds = json.load(f)

    if not preds:
        return None

    coco_dt = coco_gt.loadRes(preds)

    e = COCOeval(
        coco_gt,
        coco_dt,
        "segm",
    )

    if cat_ids is not None:
        e.params.catIds = cat_ids

    e.evaluate()
    e.accumulate()

    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()

    return e


rows = []

for res in ["640", "1024"]:

    gt = WORK_DIR / f"gt_{res}_test.json"

    for arch in MODELS:

        dt = (
            PREDICTION_DIR
            / f"{arch}_{res}_test.json"
        )

        if not dt.exists():

            for cls in ["All", "IM", "OM"]:

                rows.append(
                    dict(
                        model=arch,
                        res=int(res),
                        cls=cls,
                        mAP50=np.nan,
                        mAP50_95=np.nan,
                        mAP75=np.nan,
                        AR100=np.nan,
                        n_pred=np.nan,
                        note="checkpoint/prediction not available",
                    )
                )

            continue

        with dt.open("r", encoding="utf-8") as f:
            n_pred = len(json.load(f))

        with contextlib.redirect_stdout(io.StringIO()):

            for cls, cid in [
                ("All", None),
                ("IM", [1]),
                ("OM", [2]),
            ]:

                e = evaluate(
                    gt,
                    dt,
                    cid,
                )

                rows.append(
                    dict(
                        model=arch,
                        res=int(res),
                        cls=cls,
                        mAP50=round(
                            float(e.stats[1]),
                            4,
                        ),
                        mAP50_95=round(
                            float(e.stats[0]),
                            4,
                        ),
                        mAP75=round(
                            float(e.stats[2]),
                            4,
                        ),
                        AR100=round(
                            float(e.stats[8]),
                            4,
                        ),
                        n_pred=n_pred,
                        note="",
                    )
                )

        print(
            f"{arch:11s} {res:>4s}  "
            f"n_pred={n_pred:5d}  "
            f"AP50(All)={rows[-3]['mAP50']:.3f}  "
            f"IM={rows[-2]['mAP50']:.3f}  "
            f"OM={rows[-1]['mAP50']:.3f}",
            flush=True,
        )


# ============================================================
# SAVE UNIFIED RESULTS
# ============================================================

df = pd.DataFrame(rows)
results_file = (OUTPUT_DIR/ "06_unified_map50.csv")
df.to_csv(results_file,index=False,)

with (OUTPUT_DIR / "06_eval_params.json").open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(params, f, indent=1)


# ============================================================
# SAVE COCO EVALUATION PARAMETERS
# ============================================================

reference_gt = (
    WORK_DIR
    / "gt_640_test.json"
)

reference_pred = (
    PREDICTION_DIR
    / "YOLOv11_640_test.json"
)

if reference_pred.exists():

    e = evaluate(
        reference_gt,
        reference_pred,
    )

    p = e.params

    params = dict(

        iouType="segm",

        ap50_source=(
            "COCOeval.stats[1] "
            "(IoU=0.50, area=all, maxDets=100)"
        ),

        iouThrs=[
            round(float(x), 2)
            for x in p.iouThrs
        ],

        recThrs=(
            f"{len(p.recThrs)} points 0..1"
        ),

        maxDets=[
            int(x)
            for x in p.maxDets
        ],

        areaRng=[
            [
                float(a),
                float(b),
            ]
            for a, b in p.areaRng
        ],

        areaRngLbl=list(
            p.areaRngLbl
        ),

        useCats=int(
            p.useCats
        ),

        ground_truth=(
            "_annotations.coco.json with "
            "supercategory id 0 removed; "
            "IM=1, OM=2"
        ),

        gt_stats=(
            "52 images, 114 instances "
            "(57 IM / 57 OM)"
        ),

        per_model_inference=dict(

            YOLOv11=(
                "ultralytics predict, "
                "imgsz=<res>, conf=0.001, "
                "iou=0.7, max_det=300, "
                "retina_masks=True"
            ),

            YOLO26=(
                "ultralytics predict, "
                "imgsz=<res>, conf=0.001, "
                "iou=0.7, max_det=300, "
                "retina_masks=True"
            ),

            Detectron2=(
                "mask_rcnn_R_50_FPN_3x, "
                "NUM_CLASSES=2, "
                "MIN/MAX_SIZE_TEST=<res>, "
                "SCORE_THRESH_TEST=0.001"
            ),

            UNet=(
                "semantic -> instances by "
                "cv2.connectedComponents on "
                "sigmoid(logits)>0.5 per channel "
                "(0=IM, 1=OM); components <10 px "
                "dropped; score = mean sigmoid "
                "probability inside the component"
            ),

            SAM3=(
                "see prediction-generation script"
            ),
        ),

        notes=[
            (
                "Identical COCOeval parameters "
                "applied to every model."
            ),
            (
                "No hand-indexing of "
                "eval['precision']; AP50 read "
                "from stats[1]."
            ),
            (
                "U-Net instance derivation is "
                "a documented reconstruction, "
                "not a native output."
            ),
        ],
    )

    params_file = (
        OUTPUT_DIR
        / "03_eval_params.json"
    )

    with params_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            params,
            f,
            indent=1,
        )


print(
    "\nwrote:",
    results_file,
)

print(
    "\n=== All-class AP50 ==="
)

print(
    df[
        df.cls == "All"
    ][
        [
            "model",
            "res",
            "mAP50",
            "mAP50_95",
            "n_pred",
        ]
    ].to_string(
        index=False
    )
)