"""Final self-audit. Re-derive every headline claim and compare to what the
findings document says. Two kinds of check:

  [CSV]    claim vs the deliverable CSV it came from - catches transcription error
  [RECOMP] claim vs a fresh independent recomputation from raw data - catches
           method error, which is what actually bit twice

Any MISMATCH is printed loudly and written to the report.

Adapted from original 29_verify_claims.py.
"""

import json
import itertools
import io
import contextlib
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
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
    DATA_DIR,
    WORK_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
)


# ============================================================
# ORIGINAL 2048 COCO DATASET
# ============================================================
#
# These are the original 2048 x 2048 COCO annotations.
# They are used only for the dataset-level overlap/touching audit.
#
# Adjust COCO_2048_DIR only if the directory name in data/ differs.
#
# Expected structure:
#
# data/
# └── 2048p-COCO/
#     ├── train/
#     │   └── _annotations.coco.json
#     ├── valid/
#     │   └── _annotations.coco.json
#     └── test/
#         └── _annotations.coco.json
#
# The released dataset manifest is also treated as source data rather
# than as an evaluation output.

COCO_2048_DIR = DATA_DIR / "2048p-COCO"
MANIFEST = DATA_DIR / "03_manifest.csv"


R = []


def chk(phase, claim, stated, got, kind, tol=None, src=""):
    if stated is None:
        ok = None
    elif isinstance(stated, str) or isinstance(got, str):
        ok = str(stated) == str(got)
    else:
        t = tol if tol is not None else (
            0.0005 if abs(stated) < 10 else 0.51
        )
        ok = bool(abs(float(stated) - float(got)) <= t)

    R.append(
        dict(
            phase=phase,
            claim=claim,
            stated=stated,
            recomputed=got,
            kind=kind,
            match=(
                "-"
                if ok is None
                else ("OK" if ok else "MISMATCH")
            ),
            source=src,
        )
    )


# ============================================================
# RECOMP: annotation geometry (sec 5)
# ============================================================

ws = []
frs = []
fgs = []

for res in ["640", "1024"]:

    c = COCO(str(WORK_DIR / f"gt_{res}.json"))

    for iid in c.getImgIds():

        H = c.imgs[iid]["height"]
        W = c.imgs[iid]["width"]

        fg = np.zeros((H, W), bool)

        for a in c.loadAnns(c.getAnnIds(imgIds=iid)):

            m = c.annToMask(a).astype(bool)
            fg |= m

            ys, xs = np.nonzero(m)

            bb = (
                (ys.max() - ys.min() + 1)
                * (xs.max() - xs.min() + 1)
            )

            dt = ndimage.distance_transform_edt(m)
            sk = skeletonize(m)

            ws.append(
                (
                    int(res),
                    2 * float(np.median(dt[sk]))
                    if sk.any()
                    else np.nan,
                )
            )

            frs.append(m.sum() / bb)

        fgs.append(fg.sum() / (H * W))


wdf = pd.DataFrame(ws, columns=["res", "w"])

chk(
    "5",
    "n test annotations",
    228,
    len(frs),
    "RECOMP",
    0,
)

chk(
    "5",
    "median fill ratio",
    0.759,
    round(float(np.median(frs)), 3),
    "RECOMP",
    0.0015,
)

chk(
    "5",
    "median width px @640",
    90,
    round(float(wdf[wdf.res == 640].w.median())),
    "RECOMP",
    1,
)

chk(
    "5",
    "median width px @1024",
    142,
    round(float(wdf[wdf.res == 1024].w.median())),
    "RECOMP",
    1,
)

chk(
    "5",
    "annotations <=5px wide",
    0,
    int((wdf.w <= 5).sum()),
    "RECOMP",
    0,
)

chk(
    "5",
    "annotations fill<0.30",
    0,
    int((np.array(frs) < 0.30).sum()),
    "RECOMP",
    0,
)

chk(
    "5",
    "median foreground frac",
    0.041,
    round(float(np.median(fgs)), 3),
    "RECOMP",
    0.0015,
)


# ============================================================
# RECOMP: null baselines (sec 5)
# ============================================================

def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 0


db = []
de = []
dh = []

for res in ["640", "1024"]:

    c = COCO(str(WORK_DIR / f"gt_{res}.json"))

    for iid in c.getImgIds():

        H = c.imgs[iid]["height"]
        W = c.imgs[iid]["width"]

        for a in c.loadAnns(c.getAnnIds(imgIds=iid)):

            g = c.annToMask(a).astype(bool)

            ys, xs = np.nonzero(g)

            bb = np.zeros((H, W), bool)

            bb[
                ys.min():ys.max() + 1,
                xs.min():xs.max() + 1
            ] = True

            el = np.zeros((H, W), np.uint8)

            cv2.ellipse(
                el,
                (
                    (
                        int((xs.min() + xs.max()) / 2),
                        int((ys.min() + ys.max()) / 2),
                    ),
                    (
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    ),
                    0,
                ),
                1,
                -1,
            )

            cnts, _ = cv2.findContours(
                g.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            hu = np.zeros((H, W), np.uint8)

            cv2.fillPoly(
                hu,
                [cv2.convexHull(np.vstack(cnts))],
                1,
            )

            if res == "640":

                db.append(dice(bb, g))
                de.append(dice(el.astype(bool), g))
                dh.append(dice(hu.astype(bool), g))


chk(
    "5",
    "bbox-fill Dice @640",
    0.845,
    round(float(np.mean(db)), 3),
    "RECOMP",
    0.0015,
)

chk(
    "5",
    "ellipse Dice @640",
    0.859,
    round(float(np.mean(de)), 3),
    "RECOMP",
    0.0015,
)

chk(
    "5",
    "convex-hull Dice @640",
    0.991,
    round(float(np.mean(dh)), 3),
    "RECOMP",
    0.0015,
)


# ============================================================
# RECOMP: overlap / touching (sec 11)
# THIS IS WHAT WAS WRONG BEFORE
# ============================================================

def mindist(a, b):

    if (a & b).any():
        return 0.0

    return float(
        ndimage.distance_transform_edt(~a)[b].min()
    )


ov = 0
to = 0
pairs = 0
arr_t = {}


# Released dataset manifest.
man = pd.read_csv(
    MANIFEST,
    dtype={"parent_key": str},
)

man = (
    man[
        man.export == "2048p-COCO"
    ]
    .drop_duplicates("tile_filename")
    .set_index("tile_filename")
)


ntiles = 0
ncells = 0


for sp in ["train", "valid", "test"]:

    # Read the original released 2048 COCO annotations directly.
    gt2048 = (
        COCO_2048_DIR
        / sp
        / "_annotations.coco.json"
    )

    c = COCO(str(gt2048))

    for iid in c.getImgIds():

        ntiles += 1

        om = [
            c.annToMask(a).astype(bool)
            for a in c.loadAnns(
                c.getAnnIds(imgIds=iid)
            )
            if a["category_id"] == 2
        ]

        ncells += len(om)

        fn = c.imgs[iid]["file_name"]

        ar = (
            man.loc[fn].array_id
            if fn in man.index
            else "?"
        )

        for i, j in itertools.combinations(
            range(len(om)), 2
        ):

            pairs += 1

            d = mindist(
                om[i],
                om[j],
            )

            if (om[i] & om[j]).any():
                ov += 1

            if d <= 15:
                to += 1
                arr_t[ar] = arr_t.get(ar, 0) + 1


chk(
    "11",
    "unaugmented tiles",
    342,
    ntiles,
    "RECOMP",
    0,
)

chk(
    "11",
    "OM cells",
    382,
    ncells,
    "RECOMP",
    0,
)

chk(
    "11",
    "OM-OM pairs",
    47,
    pairs,
    "RECOMP",
    0,
)

chk(
    "11",
    "OVERLAPPING pairs",
    0,
    ov,
    "RECOMP",
    0,
)

chk(
    "11",
    "touching pairs (<=15px)",
    33,
    to,
    "RECOMP",
    0,
)

chk(
    "11",
    "arrays with >=1 touching pair",
    11,
    len(arr_t),
    "RECOMP",
    0,
)

chk(
    "11",
    "zero-touching 3-array splits",
    56,
    sum(
        1
        for cmb in itertools.combinations(
            sorted(set(man.array_id)),
            3,
        )
        if sum(
            arr_t.get(a, 0)
            for a in cmb
        ) == 0
    ),
    "RECOMP",
    0,
)


# ============================================================
# RECOMP: YOLO OM width + ring closure (sec 12)
# ============================================================

NM = {
    "640": 2.128 * 2048 / 640,
    "1024": 2.128 * 2048 / 1024,
}


def kern(r):
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * r + 1, 2 * r + 1),
    )


for res in ["640", "1024"]:

    wl = []

    with open(
        PREDICTION_DIR
        / f"YOLOv11_{res}_test.json"
    ) as fh:
        predictions = json.load(fh)

    for x in predictions:

        if (
            x["category_id"] != 2
            or x["score"] < 0.5
        ):
            continue

        b = mask_utils.decode(
            x["segmentation"]
        ).astype(bool)

        if b.sum() < 10:
            continue

        dt = ndimage.distance_transform_edt(b)
        sk = skeletonize(b)

        if sk.any():
            wl.append(
                2 * float(np.median(dt[sk]))
            )

    med = float(np.median(wl))

    chk(
        "12",
        f"YOLOv11 OM width px @{res}",
        4.00 if res == "640" else 6.32,
        round(med, 2),
        "RECOMP",
        0.02,
    )

    chk(
        "12",
        f"YOLOv11 OM width nm @{res}",
        27.2 if res == "640" else 26.9,
        round(med * NM[res], 1),
        "RECOMP",
        0.15,
    )


cl_nt = 0
cl_t = 0
n_nt = 0
n_t = 0

iou_after = []


for res in ["640", "1024"]:

    c = COCO(
        str(WORK_DIR / f"gt_{res}.json")
    )

    for mdl in ["YOLOv11", "YOLO26"]:

        P = {}

        with open(
            PREDICTION_DIR
            / f"{mdl}_{res}_test.json"
        ) as fh:
            predictions = json.load(fh)

        for x in predictions:

            if (
                x["category_id"] == 2
                and x["score"] >= 0.5
            ):
                P.setdefault(
                    x["image_id"], []
                ).append(x)

        for iid in c.getImgIds():

            H = c.imgs[iid]["height"]
            W = c.imgs[iid]["width"]

            preds = [
                mask_utils.decode(
                    x["segmentation"]
                ).astype(bool)
                for x in P.get(iid, [])
            ]

            if not preds:
                continue

            for a in c.loadAnns(
                c.getAnnIds(imgIds=iid)
            ):

                if a["category_id"] != 2:
                    continue

                g = c.annToMask(a).astype(bool)

                ys, xs = np.nonzero(g)

                tr = bool(
                    ys.min() == 0
                    or xs.min() == 0
                    or ys.max() == H - 1
                    or xs.max() == W - 1
                )

                edge = (
                    g
                    & ~cv2.erode(
                        g.astype(np.uint8),
                        kern(1),
                    ).astype(bool)
                )

                dg = (
                    ndimage
                    .distance_transform_edt(~edge)
                )

                bp = min(
                    preds,
                    key=lambda p:
                    dg[p].mean()
                    if p.any()
                    else 1e9,
                )

                f = ndimage.binary_fill_holes(
                    cv2.morphologyEx(
                        bp.astype(np.uint8),
                        cv2.MORPH_CLOSE,
                        kern(5),
                    ).astype(bool)
                )

                closed = (
                    f.sum()
                    >= 2 * max(bp.sum(), 1)
                )

                if tr:

                    n_t += 1
                    cl_t += closed

                else:

                    n_nt += 1
                    cl_nt += closed

                    if closed:

                        iou_after.append(
                            (f & g).sum()
                            / max(
                                (f | g).sum(),
                                1,
                            )
                        )


chk(
    "12",
    "ring closure, not truncated",
    0.53,
    round(cl_nt / n_nt, 3),
    "RECOMP",
    0.006,
)

chk(
    "12",
    "ring closure, truncated",
    0.00,
    round(
        cl_t / max(n_t, 1),
        3,
    ),
    "RECOMP",
    0.0005,
)

chk(
    "12",
    "IoU after fill among closed",
    0.978,
    round(
        float(np.mean(iou_after)),
        3,
    ),
    "RECOMP",
    0.0015,
)


# ============================================================
# RECOMP: spot-check two AP50 values (sec 3)
# ============================================================

def ap50(gtj, f, cat=None):

    with open(f) as fh:
        preds = [
            dict(x)
            for x in json.load(fh)
        ]

    gt = COCO(str(gtj))

    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        dt = gt.loadRes(preds)

        e = COCOeval(
            gt,
            dt,
            "segm",
        )

        if cat is not None:
            e.params.catIds = [cat]

        e.evaluate()
        e.accumulate()
        e.summarize()

    return float(e.stats[1])


chk(
    "3",
    "SAM3 640 AP50 All",
    0.9806,
    round(
        ap50(
            WORK_DIR / "gt_640.json",
            PREDICTION_DIR
            / "SAM3_640_test.json",
        ),
        4,
    ),
    "RECOMP",
    0.0002,
)

chk(
    "3",
    "YOLOv11 640 AP50 OM",
    0.0000,
    round(
        ap50(
            WORK_DIR / "gt_640.json",
            PREDICTION_DIR
            / "YOLOv11_640_test.json",
            2,
        ),
        4,
    ),
    "RECOMP",
    0.0002,
)


# ============================================================
# CSV consistency for the rest
# ============================================================

u = pd.read_csv(
    OUTPUT_DIR / "06_unified_map50.csv"
)

u = (
    u[u.cls == "All"]
    .set_index(["model", "res"])
    .mAP50
)


for (m, r), v in {

    ("SAM3", 640): 0.981,
    ("U-Net", 640): 0.935,
    ("Detectron2", 640): 0.906,
    ("YOLOv11", 640): 0.491,
    ("YOLO26", 640): 0.471,

    ("SAM3", 1024): 0.988,
    ("U-Net", 1024): 0.912,
    ("Detectron2", 1024): 0.482,
    ("YOLOv11", 1024): 0.484,
    ("YOLO26", 1024): 0.479,

}.items():

    chk(
        "3",
        f"unified AP50 {m} {r}",
        v,
        round(
            float(u.loc[(m, r)]),
            3,
        ),
        "CSV",
        0.0015,
        "06_unified_map50.csv",
    )


t = pd.read_csv(
    OUTPUT_DIR
    / "07_threshold_comparison.csv"
)


chk(
    "4",
    "mean delta_f1 (test vs valid)",
    0.114,
    round(
        float(t.delta_f1.mean()),
        3,
    ),
    "CSV",
    0.0015,
    "07_threshold_comparison.csv",
)

chk(
    "4",
    "mean delta vs fixed 0.5",
    0.034,
    round(
        float(
            t.delta_f1_vs_fixed.mean()
        ),
        3,
    ),
    "CSV",
    0.0015,
    "07_threshold_comparison.csv",
)

chk(
    "4",
    "rows valid-thr worse than fixed",
    25,
    int(
        (
            t.f1_at_valid_thr
            < t.f1_at_fixed_0p5
        ).sum()
    ),
    "CSV",
    0,
    "07_threshold_comparison.csv",
)


# ============================================================
# ORIGINAL DEPENDENCY:
# 05_detectron2_audit.json
#
# Producer was not present among the 13 supplied scripts.
# Retained because original 29_verify_claims.py checks it.
# ============================================================

with open(
    OUTPUT_DIR / "05_detectron2_audit.json"
) as fh:
    d2 = json.load(fh)


chk(
    "2",
    "D2 640 COCOeval AP50",
    0.906,
    round(
        d2["640"]["coco"]["AP50"] / 100,
        3,
    ),
    "CSV",
    0.0015,
    "05_detectron2_audit.json",
)

chk(
    "2",
    "D2 1024 COCOeval AP50",
    0.482,
    round(
        d2["1024"]["coco"]["AP50"] / 100,
        3,
    ),
    "CSV",
    0.0015,
    "05_detectron2_audit.json",
)

chk(
    "2",
    "D2 1024 APm",
    0.079,
    round(
        d2["1024"]["coco"]["APm"] / 100,
        3,
    ),
    "CSV",
    0.0015,
    "05_detectron2_audit.json",
)

chk(
    "2",
    "D2 640 sklearn pixel-AP",
    0.800,
    round(
        d2["640"]["sklearn"]["all"][
            "pixel_ap"
        ],
        3,
    ),
    "CSV",
    0.0015,
    "05_detectron2_audit.json",
)


# ============================================================
# Boundary metrics
# ============================================================

b = pd.read_csv(
    OUTPUT_DIR
    / "10_boundary_metrics_summary.csv"
)

g = b.set_index(
    ["model", "res", "cls"]
)


chk(
    "12",
    "YOLOv11 640 OM SSD nm",
    21.0,
    round(
        float(
            g.loc[
                ("YOLOv11", 640, "OM")
            ].ssd_nm
        ),
        1,
    ),
    "CSV",
    0.06,
    "10_boundary_metrics_summary.csv",
)

chk(
    "12",
    "SAM3 640 OM SSD nm",
    14.3,
    round(
        float(
            g.loc[
                ("SAM3", 640, "OM")
            ].ssd_nm
        ),
        1,
    ),
    "CSV",
    0.06,
    "10_boundary_metrics_summary.csv",
)


# ============================================================
# Fill diagnostic
# ============================================================

f = pd.read_csv(
    OUTPUT_DIR
    / "10_fill_and_score.csv"
)

f = f.set_index(
    ["model", "res", "cls"]
)


chk(
    "12",
    "YOLOv11 640 OM AP50 after fill",
    0.138,
    round(
        float(
            f.loc[
                ("YOLOv11", 640, "OM")
            ].ap50_after
        ),
        3,
    ),
    "CSV",
    0.0015,
    "10_fill_and_score.csv",
)

chk(
    "12",
    "SAM3 640 fill delta (control)",
    0.000,
    round(
        float(
            f.loc[
                ("SAM3", 640, "All")
            ].delta
        ),
        4,
    ),
    "CSV",
    0.0002,
    "10_fill_and_score.csv",
)


# ============================================================
# ORIGINAL DEPENDENCY:
# 08c_breakage_at_rebuttal_point.csv
#
# Producer was not present among the 13 supplied scripts.
# Retained because original 29_verify_claims.py checks it.
# ============================================================

ct = pd.read_csv(
    OUTPUT_DIR
    / "08c_breakage_at_rebuttal_point.csv"
).set_index(
    ["model", "res", "cls"]
)


chk(
    "6",
    "YOLOv11 640 breakage @rebuttal pt",
    0.278,
    round(
        float(
            ct.loc[
                ("YOLOv11", 640, "All")
            ].breakage_rate
        ),
        3,
    ),
    "CSV",
    0.0015,
    "08c_breakage_at_rebuttal_point.csv",
)

chk(
    "6",
    "SAM3 640 breakage @rebuttal pt",
    0.009,
    round(
        float(
            ct.loc[
                ("SAM3", 640, "All")
            ].breakage_rate
        ),
        3,
    ),
    "CSV",
    0.0015,
    "08c_breakage_at_rebuttal_point.csv",
)


# ============================================================
# Write final claim-verification report
# ============================================================

df = pd.DataFrame(R)

df.to_csv(
    OUTPUT_DIR
    / "11_claim_verification.csv",
    index=False,
)


bad = df[
    df.match == "MISMATCH"
]


print(
    f"checks: {len(df)}  "
    f"OK: {int((df.match == 'OK').sum())}  "
    f"MISMATCH: {len(bad)}"
)

print(
    "  independently recomputed: "
    f"{int((df.kind == 'RECOMP').sum())}   "
    "CSV-consistency: "
    f"{int((df.kind == 'CSV').sum())}"
)


if len(bad):

    print(
        "\n!!! MISMATCHES !!!"
    )

    print(
        bad[
            [
                "phase",
                "claim",
                "stated",
                "recomputed",
                "kind",
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nno mismatches"
    )


print(
    "\n--- all checks ---"
)

print(
    df[
        [
            "phase",
            "claim",
            "stated",
            "recomputed",
            "kind",
            "match",
        ]
    ].to_string(index=False)
)