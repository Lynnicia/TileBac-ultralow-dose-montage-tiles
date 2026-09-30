"""Phase 3.1 - dump raw predictions for every model in COCO detection format.

One record per predicted instance:
    {"image_id": int, "category_id": int, "segmentation": <RLE>, "score": float}

Category ids follow the COCO GT file: IM=1, OM=2.
All models write to out/predictions/<arch>_<res>_test.json.
"""
import json, os, sys, glob
import numpy as np, torch, cv2
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = "/home/vnw/tilebac_reanalysis"
OUT, WORK = f"{ROOT}/out", f"{ROOT}/work"
PRED = f"{OUT}/predictions"; os.makedirs(PRED, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANN = {"640":  f"{ROOT}/data/uldm_bm/annotated/640p-COCO/BCET Montage 2048-Tile.v9i.coco-segmentation/test",
       "1024": f"{ROOT}/data/uldm_bm/annotated/1024p-COCO/BCET Montage 2048-Tile.v8i.coco-segmentation/test"}

def gt_path(res):
    """Canonical GT: supercategory id 0 dropped, IM=1, OM=2. Written to work/."""
    p = f"{WORK}/gt_{res}.json"
    if not os.path.exists(p):
        j = json.load(open(f"{ANN[res]}/_annotations.coco.json"))
        j["categories"] = [c for c in j["categories"] if c["id"] != 0]
        json.dump(j, open(p, "w"))
    return p

def encode(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r

def save(preds, arch, res):
    p = f"{PRED}/{arch}_{res}_test.json"
    json.dump(preds, open(p, "w"))
    print(f"  -> {p}  ({len(preds)} instances)", flush=True)

def images(res):
    coco = COCO(gt_path(res))
    return coco, [(i, coco.imgs[i]) for i in sorted(coco.getImgIds())]

#Yolo
    def run_yolo(arch, res, ckpt):
    from ultralytics import YOLO
    model = YOLO(ckpt)
    coco, imgs = images(res)
    preds = []
    for img_id, info in imgs:
        fp = os.path.join(ANN[res], info["file_name"])
        r = model.predict(fp, imgsz=int(res), conf=0.001, iou=0.7, max_det=300,
                          retina_masks=True, verbose=False, device=0)[0]
        if r.masks is None:
            continue
        H, W = info["height"], info["width"]
        for m, c, s in zip(r.masks.data.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.conf.cpu().numpy()):
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            preds.append(dict(image_id=int(img_id), category_id=int(c) + 1,
                              segmentation=encode(m > 0.5), score=float(s)))
    save(preds, arch, res)

# Detect2
def run_d2(res, ckpt):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = ckpt
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.INPUT.MIN_SIZE_TEST = int(res); cfg.INPUT.MAX_SIZE_TEST = int(res)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    model = build_model(cfg); DetectionCheckpointer(model).load(ckpt)
    model.to(DEV).eval()
    coco, imgs = images(res)
    preds = []
    with torch.no_grad():
        for img_id, info in imgs:
            im = cv2.imread(os.path.join(ANN[res], info["file_name"]))
            H, W = im.shape[:2]
            out = model([{"image": torch.as_tensor(im.transpose(2, 0, 1)).float().to(DEV),
                          "height": H, "width": W}])[0]["instances"].to("cpu")
            for i in range(len(out)):
                preds.append(dict(image_id=int(img_id),
                                  category_id=int(out.pred_classes[i].item()) + 1,
                                  segmentation=encode(out.pred_masks[i].numpy()),
                                  score=float(out.scores[i].item())))
    save(preds, "Detectron2", res)

#  U-Net
def run_unet(res, ckpt):
    """Semantic -> instances via connected components (documented derivation)."""
    # load model.py directly: the repo's src/__init__.py is not working
    import importlib.util
    _mp = f"{ROOT}/repos/Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets/src/model.py"
    _spec = importlib.util.spec_from_file_location("unet_model", _mp)
    _mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    UNet = _mod.UNet
    from PIL import Image
    import torchvision.transforms as T
    model = UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(DEV).eval()
    coco, imgs = images(res)
    preds = []
    with torch.no_grad():
        for img_id, info in imgs:
            im = Image.open(os.path.join(ANN[res], info["file_name"])).convert("L")
            x = T.ToTensor()(im).unsqueeze(0).to(DEV)
            prob = torch.sigmoid(model(x))[0].cpu().numpy()      # [2,H,W]; sigmoid applied
            for ch in (0, 1):                                     # 0=IM, 1=OM 
                binm = (prob[ch] > 0.5).astype(np.uint8)
                n, lab = cv2.connectedComponents(binm)
                for k in range(1, n):
                    comp = lab == k
                    if comp.sum() < 10:                           # drop specks in img
                        continue
                    preds.append(dict(image_id=int(img_id), category_id=ch + 1,
                                      segmentation=encode(comp),
                                      score=float(prob[ch][comp].mean())))
    save(preds, "U-Net", res)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for res in ["640", "1024"]:
        if which in ("all", "yolo"):
            print(f"[YOLOv11 {res}]", flush=True); run_yolo("YOLOv11", res, f"{ROOT}/ckpt/{res}-YOLOv11-ULDM.pt")
            print(f"[YOLO26 {res}]", flush=True);  run_yolo("YOLO26", res, f"{ROOT}/ckpt/{res}-YOLO26-ULDM.pt")
        if which in ("all", "d2"):
            print(f"[Detectron2 {res}]", flush=True); run_d2(res, f"{ROOT}/ckpt/{res}-Detectron2-ULDM.pth")
        if which in ("all", "unet"):
            print(f"[U-Net {res}]", flush=True); run_unet(res, f"{ROOT}/ckpt/{res}-U-Net-ULDM.pt")
