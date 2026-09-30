"""Phase 10 support - how wide is each model's predicted structure?

Same local-width measure used on the annotations in Phase 6: 2 x distance-to-
background, median over the mask's skeleton. helps answer whether YOLO's OM output is a
thin trace and, if so, how thin.
"""
import json
import numpy as np, pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize
from pycocotools import mask as mask_utils

ROOT="/home/vnw/tilebac_reanalysis"; OUT=f"{ROOT}/out"; PRED=f"{OUT}/predictions"
MODELS=["YOLOv11","YOLO26","U-Net","Detectron2","SAM3"]; CLS={1:"IM",2:"OM"}
NM={"640":2.128*2048/640,"1024":2.128*2048/1024}
rows=[]
for res in ["640","1024"]:
    for m in MODELS:
        for x in json.load(open(f"{PRED}/{m}_{res}_test.json")):
            if x["score"]<0.5: continue
            b=mask_utils.decode(x["segmentation"]).astype(bool)
            if b.sum()<10: continue
            dt=ndimage.distance_transform_edt(b); sk=skeletonize(b)
            w=2*float(np.median(dt[sk])) if sk.any() else np.nan
            ys,xs=np.nonzero(b)
            bb=(ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)
            rows.append(dict(model=m,res=int(res),cls=CLS[x["category_id"]],
                score=round(float(x["score"]),4),area=int(b.sum()),
                median_width_px=round(w,2),median_width_nm=round(w*NM[res],1),
                fill_ratio=round(b.sum()/bb,4)))
df=pd.DataFrame(rows); df.to_csv(f"{OUT}/10_pred_width.csv",index=False)
s=df.groupby(["model","res","cls"]).agg(n=("median_width_px","size"),
    width_px=("median_width_px","median"),width_nm=("median_width_nm","median"),
    fill=("fill_ratio","median"),area=("area","median")).round(2)
s.to_csv(f"{OUT}/10_pred_width_summary.csv")
print(s.to_string())
