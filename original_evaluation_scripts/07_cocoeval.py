#mainfix for coco eval pararmaters 
"""Phase 3.2 - one COCOeval pass, identical parameters, every model.

ap50 = coco_eval.stats[1]  (never hand-indexed from eval["precision"])
"""
import json, os, glob, io, contextlib
import numpy as np, pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = "/home/vnw/tilebac_reanalysis" #tied to my directory ondgcx 
OUT, WORK = f"{ROOT}/out", f"{ROOT}/work"
PRED = f"{OUT}/predictions"
CATS = {1: "IM", 2: "OM"}

def evaluate(gt_json, dt_json, cat_ids=None):
    coco_gt = COCO(gt_json)
    preds = json.load(open(dt_json))
    if not preds:
        return None
    coco_dt = coco_gt.loadRes(preds)
    e = COCOeval(coco_gt, coco_dt, "segm")
    if cat_ids is not None:
        e.params.catIds = cat_ids
    e.evaluate(); e.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        e.summarize()
    return e

rows = []
for res in ["640", "1024"]:
    gt = f"{WORK}/gt_{res}.json"
    for arch in ["YOLOv11", "YOLO26", "U-Net", "Detectron2", "SAM3"]:
        dt = f"{PRED}/{arch}_{res}_test.json"
        if not os.path.exists(dt):
            for cls in ["All", "IM", "OM"]:
                rows.append(dict(model=arch, res=int(res), cls=cls, mAP50=np.nan,
                                 mAP50_95=np.nan, mAP75=np.nan, AR100=np.nan,
                                 n_pred=np.nan,
                                 note="checkpoint/prediction not available"))
            continue
        n_pred = len(json.load(open(dt)))
        with contextlib.redirect_stdout(io.StringIO()):
            for cls, cid in [("All", None), ("IM", [1]), ("OM", [2])]:
                e = evaluate(gt, dt, cid)
                rows.append(dict(model=arch, res=int(res), cls=cls,
                                 mAP50=round(float(e.stats[1]), 4),
                                 mAP50_95=round(float(e.stats[0]), 4),
                                 mAP75=round(float(e.stats[2]), 4),
                                 AR100=round(float(e.stats[8]), 4),
                                 n_pred=n_pred, note=""))
        print(f"{arch:11s} {res:>4s}  n_pred={n_pred:5d}  "
              f"AP50(All)={rows[-3]['mAP50']:.3f}  IM={rows[-2]['mAP50']:.3f}  "
              f"OM={rows[-1]['mAP50']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/06_unified_map50.csv", index=False)

# eval params sidecar
e = evaluate(f"{WORK}/gt_640.json", f"{PRED}/YOLOv11_640_test.json")
p = e.params
params = dict(
    iouType="segm",
    ap50_source="COCOeval.stats[1] (IoU=0.50, area=all, maxDets=100)",
    iouThrs=[round(float(x), 2) for x in p.iouThrs],
    recThrs=f"{len(p.recThrs)} points 0..1",
    maxDets=[int(x) for x in p.maxDets],
    areaRng=[[float(a), float(b)] for a, b in p.areaRng],
    areaRngLbl=list(p.areaRngLbl),
    useCats=int(p.useCats),
    ground_truth="_annotations.coco.json with supercategory id 0 removed; IM=1, OM=2",
    gt_stats="52 images, 114 instances (57 IM / 57 OM)",
    per_model_inference=dict(
        YOLOv11="ultralytics predict, imgsz=<res>, conf=0.001, iou=0.7, max_det=300, retina_masks=True",
        YOLO26="ultralytics predict, imgsz=<res>, conf=0.001, iou=0.7, max_det=300, retina_masks=True",
        Detectron2="mask_rcnn_R_50_FPN_3x, NUM_CLASSES=2, MIN/MAX_SIZE_TEST=<res>, SCORE_THRESH_TEST=0.001",
        UNet=("semantic -> instances by cv2.connectedComponents on sigmoid(logits)>0.5 "
              "per channel (0=IM, 1=OM); components <10 px dropped; "
              "score = mean sigmoid probability inside the component"),
        SAM3="see note in 06_unified_map50.csv"),
    notes=["Identical COCOeval parameters applied to every model.",
           "No hand-indexing of eval['precision']; AP50 read from stats[1].",
           "U-Net instance derivation is a documented reconstruction, not a native output."],
    torch="2.5.1+cu121", detectron2="0.6", ultralytics="8.4.122", device="Tesla V100-SXM3-32GB")
json.dump(params, open(f"{OUT}/06_eval_params.json", "w"), indent=1)
print("\nwrote out/06_unified_map50.csv and out/06_eval_params.json")
print("\n=== All-class AP50 ===")
print(df[df.cls == "All"][["model", "res", "mAP50", "mAP50_95", "n_pred"]].to_string(index=False))
