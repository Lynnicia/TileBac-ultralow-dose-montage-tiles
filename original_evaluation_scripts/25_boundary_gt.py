"""Phase 10.2 - score every model against a BOUNDARY ground truth.

The released GT is a filled region. Here each GT polygon is converted to a band of
width w centred on outline, and all 10 models are rescored against it with the
same COCOeval call. Sweeps w = 3, 5, 7, 10, 15 px.

If a model predicts membranes as boundaries its AP50 should rise sharply against a
boundary GT and fall against the region GT. a region-predicting model does the
opposite. This tests the convention mismatch i n the code
"""
import json, os, io, contextlib
import numpy as np, pandas as pd, cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

ROOT="/home/vnw/tilebac_reanalysis"; OUT=f"{ROOT}/out"
PRED=f"{OUT}/predictions"; WORK=f"{ROOT}/work"
MODELS=["YOLOv11","YOLO26","U-Net","Detectron2","SAM3"]; CLS={1:"IM",2:"OM"}
WIDTHS=[3,5,7,10,15]
NM={"640":2.128*2048/640, "1024":2.128*2048/1024}

def kern(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def band(m,w):
    """band of width ~w centred on the outline of m"""
    r=max(1,w//2); k=kern(r)
    u=m.astype(np.uint8)
    return (cv2.dilate(u,k).astype(bool)) & (~cv2.erode(u,k).astype(bool))
def enc(m):
    r=mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"]=r["counts"].decode("ascii"); return r

rows=[]
for res in ["640","1024"]:
    base=json.load(open(f"{WORK}/gt_{res}.json"))
    coco=COCO(f"{WORK}/gt_{res}.json")
    for w in WIDTHS:
        g=dict(images=base["images"],categories=base["categories"],annotations=[])
        for a in base["annotations"]:
            m=coco.annToMask(a).astype(bool)
            b=band(m,w)
            if not b.any(): continue
            r=enc(b)
            g["annotations"].append(dict(id=a["id"],image_id=a["image_id"],
                category_id=a["category_id"],segmentation=r,iscrowd=0,
                area=float(mask_utils.area(mask_utils.frPyObjects(r,r["size"][0],r["size"][1]))
                           if isinstance(r["counts"],list) else int(b.sum())),
                bbox=list(map(float,cv2.boundingRect(b.astype(np.uint8))))))
        p=f"{WORK}/gt_{res}_band{w}.json"; json.dump(g,open(p,"w"))
        gt=COCO(p)
        for m in MODELS:
            preds=[dict(x) for x in json.load(open(f"{PRED}/{m}_{res}_test.json"))]
            for cid,cn in [(None,"All"),(1,"IM"),(2,"OM")]:
                pp=[dict(x) for x in preds]
                with contextlib.redirect_stdout(io.StringIO()):
                    dt=gt.loadRes(pp); e=COCOeval(gt,dt,"segm")
                    if cid is not None: e.params.catIds=[cid]
                    e.evaluate(); e.accumulate(); e.summarize()
                rows.append(dict(model=m,res=int(res),cls=cn,band_width_px=w,
                    band_width_nm=round(w*NM[res],1),ap50=round(float(e.stats[1]),4),
                    ap50_95=round(float(e.stats[0]),4)))
        print(f"  res={res} w={w} done",flush=True)
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/10_boundary_gt_sweep.csv",index=False)
print("\n=== AP50 vs boundary-GT band width (All classes) ===")
print(df[df.cls=="All"].pivot_table(index=["model","res"],columns="band_width_px",values="ap50").to_string())
print("\n=== OM only ===")
print(df[df.cls=="OM"].pivot_table(index=["model","res"],columns="band_width_px",values="ap50").to_string())
