"""Phase 3.1 (YOLO) - use ultralytics' own COCO export rather than re-deriving masks.

model.val(save_json=True) writes predictions.json using ultralytics' native mask
post-processing (the same path that produces `metrics.seg.map50` in the published
scripts). Only the image_id is remapped, from Roboflow filename stem to the COCO
image id. Masks and scores are passed through untouched.
"""
import json, os, glob
from pycocotools.coco import COCO

ROOT = "/home/vnw/tilebac_reanalysis"
SRC = "/home/vnw/acorn/runs/segment/work/valjson"
PRED = f"{ROOT}/out/predictions"

for res in ["640", "1024"]:
    coco = COCO(f"{ROOT}/work/gt_{res}.json")
    stem2id = {os.path.splitext(v["file_name"])[0]: k for k, v in coco.imgs.items()}
    for arch in ["YOLOv11", "YOLO26"]:
        src = f"{SRC}/{arch}_{res}/predictions.json"
        j = json.load(open(src))
        out, miss = [], 0
        for x in j:
            iid = stem2id.get(x["image_id"])
            if iid is None:
                miss += 1; continue
            out.append(dict(image_id=int(iid), category_id=int(x["category_id"]),
                            segmentation=x["segmentation"], score=float(x["score"])))
        assert miss == 0, f"{arch} {res}: {miss} unmapped image ids"
        json.dump(out, open(f"{PRED}/{arch}_{res}_test.json", "w"))
        print(f"{arch} {res}: {len(out)} instances, 0 unmapped")
