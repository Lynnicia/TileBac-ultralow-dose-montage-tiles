"""Phase 8.4 + 8.5 - null-baseline Dice per instance, and annotation-geometry arrays."""
import json, os
import numpy as np, pandas as pd, cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

ROOT="/home/vnw/tilebac_reanalysis"; OUT=f"{ROOT}/out"; FD=f"{OUT}/figure_data"
PRED=f"{OUT}/predictions"; WORK=f"{ROOT}/work"; MDIR=f"{FD}/masks"
MODELS=["YOLOv11","YOLO26","U-Net","Detectron2","SAM3"]; CLS={1:"IM",2:"OM"}

def dice(a,b):
    s=a.sum()+b.sum(); return float(2*(a&b).sum()/s) if s else 0.0
def iou(a,b):
    u=(a|b).sum(); return float((a&b).sum()/u) if u else 0.0
def bbox_mask(g,H,W):
    ys,xs=np.nonzero(g); m=np.zeros((H,W),bool)
    m[ys.min():ys.max()+1, xs.min():xs.max()+1]=True; return m
def ellipse_mask(g,H,W):
    ys,xs=np.nonzero(g); m=np.zeros((H,W),np.uint8)
    cv2.ellipse(m,((int((xs.min()+xs.max())/2),int((ys.min()+ys.max())/2)),
                   (int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)),0),1,-1)
    return m.astype(bool)
def hull_mask(g,H,W):
    cnts,_=cv2.findContours(g.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    pts=np.vstack(cnts); h=cv2.convexHull(pts)
    m=np.zeros((H,W),np.uint8); cv2.fillPoly(m,[h],1); return m.astype(bool)

# 8.4 null
rows=[]
for res in ["640","1024"]:
    gt=COCO(f"{WORK}/gt_{res}.json")
    Pm={}
    for m in MODELS:
        d={}
        for x in json.load(open(f"{PRED}/{m}_{res}_test.json")): d.setdefault(x["image_id"],[]).append(x)
        Pm[m]=d
    for iid in sorted(gt.getImgIds()):
        info=gt.imgs[iid]; H,W=info["height"],info["width"]
        dec={m:[(x["category_id"],mask_utils.decode(x["segmentation"]).astype(bool),float(x["score"]))
                for x in Pm[m].get(iid,[]) if x["score"]>=0.5] for m in MODELS}
        for k,a in enumerate(gt.loadAnns(gt.getAnnIds(imgIds=iid)),1):
            g=gt.annToMask(a).astype(bool)
            if not g.any(): continue
            cid=a["category_id"]; cn=CLS[cid]
            base=dict(res=int(res),image_id=iid,instance=k,cls=cn,gt_area=int(g.sum()))
            for name,mm in [("null_bbox_fill",bbox_mask(g,H,W)),
                            ("null_ellipse",ellipse_mask(g,H,W)),
                            ("null_convex_hull",hull_mask(g,H,W))]:
                rows.append(dict(**base,method=name,dice=round(dice(mm,g),5),
                                 iou=round(iou(mm,g),5),kind="null"))
            for m in MODELS:
                cand=[p for c,p,s in dec[m] if c==cid]
                if cand:
                    ds=[dice(p,g) for p in cand]; j=int(np.argmax(ds))
                    rows.append(dict(**base,method=m,dice=round(ds[j],5),
                                     iou=round(iou(cand[j],g),5),kind="model"))
                else:
                    rows.append(dict(**base,method=m,dice=0.0,iou=0.0,kind="model"))
d=pd.DataFrame(rows)
d.to_csv(f"{FD}/08_null_baseline_per_instance.csv", index=False)

def ci95(x):
    x=np.asarray(x,float); n=len(x)
    if n<2: return (np.nan,np.nan)
    se=x.std(ddof=1)/np.sqrt(n); return (x.mean()-1.96*se, x.mean()+1.96*se)
srows=[]
for (res,cls,meth),g in d.groupby(["res","cls","method"]):
    lo,hi=ci95(g.dice)
    srows.append(dict(res=res,cls=cls,method=meth,kind=g.kind.iloc[0],n=len(g),
                      mean_dice=round(g.dice.mean(),5),ci95_lo=round(lo,5),ci95_hi=round(hi,5),
                      median_dice=round(g.dice.median(),5),
                      mean_iou=round(g.iou.mean(),5)))
for (res,meth),g in d.groupby(["res","method"]):
    lo,hi=ci95(g.dice)
    srows.append(dict(res=res,cls="All",method=meth,kind=g.kind.iloc[0],n=len(g),
                      mean_dice=round(g.dice.mean(),5),ci95_lo=round(lo,5),ci95_hi=round(hi,5),
                      median_dice=round(g.dice.median(),5),mean_iou=round(g.iou.mean(),5)))
s=pd.DataFrame(srows).sort_values(["res","cls","mean_dice"],ascending=[True,True,False])
s.to_csv(f"{FD}/08_null_baseline_summary.csv", index=False)
print("=== null-baseline summary (All classes) ===")
print(s[s.cls=="All"][["res","method","kind","n","mean_dice","ci95_lo","ci95_hi"]].to_string(index=False))
#8.5 geo for one tile
sel=pd.read_csv(f"{FD}/08_tile_selection.csv")
t=sel[sel.figure_category=="multi_separated"].iloc[0]
for res in ["640","1024"]:
    gt=COCO(f"{WORK}/gt_{res}.json"); iid=int(t[f"image_id_{res}"])
    info=gt.imgs[iid]; H,W=info["height"],info["width"]
    dd=f"{MDIR}/{t.figure_category}"; os.makedirs(dd,exist_ok=True)
    for cid,cn in CLS.items():
        anns=[a for a in gt.loadAnns(gt.getAnnIds(imgIds=iid)) if a["category_id"]==cid]
        for nm,fn in [("polygon",lambda g:g),("bboxfill",lambda g:bbox_mask(g,H,W)),
                      ("ellipse",lambda g:ellipse_mask(g,H,W)),("hull",lambda g:hull_mask(g,H,W))]:
            u=np.zeros((H,W),np.uint8)
            for a in anns: u|=fn(gt.annToMask(a).astype(bool)).astype(np.uint8)
            np.save(f"{dd}/geometry_{res}_{cn}_{nm}.npy", u)
print(f"\ngeometry arrays -> {MDIR}/{t.figure_category}/geometry_*.npy (co-registered, full res)")

# 8.5 distro
geo=pd.read_csv(f"{OUT}/08b_annotation_geometry.csv")
g=geo[geo.split=="test"]
g[["res","cls","image_id","area","bbox_area","fill_ratio"]].to_csv(f"{FD}/08_fill_ratio_all.csv",index=False)
g[["res","cls","image_id","median_width_px","max_width_px","area"]].to_csv(f"{FD}/08_local_width_all.csv",index=False)
print(f"08_fill_ratio_all.csv: {len(g)} rows | 08_local_width_all.csv: {len(g)} rows")
