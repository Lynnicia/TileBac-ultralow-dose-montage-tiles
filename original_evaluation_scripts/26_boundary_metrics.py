"""Phase 10.3 - boundary IoU and symmetric surface distance, all 10 models.

For each GT instance, the same-class prediction (score>=0.5) with the highest
boundary IoU is taken as match. Boundary IoU is computed on outline bands of
width BW; symmetric surface distance is the mean of the two directed mean
nearest-neighbour distances between contour pixel sets.
"""
import json, os
import numpy as np, pandas as pd, cv2
from scipy import ndimage
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

ROOT="/home/vnw/tilebac_reanalysis"; OUT=f"{ROOT}/out"
PRED=f"{OUT}/predictions"; WORK=f"{ROOT}/work"
MODELS=["YOLOv11","YOLO26","U-Net","Detectron2","SAM3"]; CLS={1:"IM",2:"OM"}
BW=5
NM={"640":2.128*2048/640, "1024":2.128*2048/1024}

def kern(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def band(m,w=BW):
    r=max(1,w//2); u=m.astype(np.uint8); k=kern(r)
    return (cv2.dilate(u,k).astype(bool)) & (~cv2.erode(u,k).astype(bool))
def contour(m):
    u=m.astype(np.uint8)
    return (u.astype(bool)) & (~cv2.erode(u,kern(1)).astype(bool))
def iou(a,b):
    u=(a|b).sum(); return float((a&b).sum()/u) if u else 0.0
def ssd(pm,gm):
    """symmetric surface distance in px between two contour pixel sets"""
    cp,cg=contour(pm),contour(gm)
    if not cp.any() or not cg.any(): return np.nan
    dg=ndimage.distance_transform_edt(~cg)     # distance to GT contour
    dp=ndimage.distance_transform_edt(~cp)
    return float(0.5*(dg[cp].mean()+dp[cg].mean()))

rows=[]
for res in ["640","1024"]:
    coco=COCO(f"{WORK}/gt_{res}.json")
    for m in MODELS:
        P={}
        for x in json.load(open(f"{PRED}/{m}_{res}_test.json")):
            if x["score"]>=0.5: P.setdefault(x["image_id"],[]).append(x)
        for iid in sorted(coco.getImgIds()):
            preds=[(x["category_id"],mask_utils.decode(x["segmentation"]).astype(bool),x["score"])
                   for x in P.get(iid,[])]
            pb=[(c,band(pm),pm,s) for c,pm,s in preds]
            for k,a in enumerate(coco.loadAnns(coco.getAnnIds(imgIds=iid)),1):
                gm=coco.annToMask(a).astype(bool); gb=band(gm)
                cid=a["category_id"]
                cand=[(iou(b,gb),pm,s) for c,b,pm,s in pb if c==cid]
                if cand:
                    biou,pm,s=max(cand,key=lambda z:z[0])
                    rows.append(dict(model=m,res=int(res),cls=CLS[cid],image_id=iid,instance=k,
                        matched=True,score=round(float(s),4),
                        boundary_iou=round(biou,4),
                        region_iou=round(iou(pm,gm),4),
                        ssd_px=round(ssd(pm,gm),3),
                        ssd_nm=round(ssd(pm,gm)*NM[res],1) if np.isfinite(ssd(pm,gm)) else np.nan,
                        pred_area=int(pm.sum()),gt_area=int(gm.sum())))
                else:
                    rows.append(dict(model=m,res=int(res),cls=CLS[cid],image_id=iid,instance=k,
                        matched=False,score=np.nan,boundary_iou=np.nan,region_iou=np.nan,
                        ssd_px=np.nan,ssd_nm=np.nan,pred_area=0,gt_area=int(gm.sum())))
        print(f"  {m} {res} done",flush=True)
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/10_boundary_metrics.csv",index=False)
s=(df[df.matched].groupby(["model","res","cls"])
   .agg(n=("boundary_iou","size"),boundary_iou=("boundary_iou","mean"),
        region_iou=("region_iou","mean"),ssd_px=("ssd_px","mean"),ssd_nm=("ssd_nm","mean")).round(4))
s.to_csv(f"{OUT}/10_boundary_metrics_summary.csv")
print("\n=== mean boundary IoU / region IoU / SSD ===")
print(s.to_string())
