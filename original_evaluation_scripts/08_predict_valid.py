"""Phase 4 support - dump predictions on the VALIDATION partition for all 10 models.

Identical inference settings to the test-partition dump (scripts 06/06b/06c).
"""
import json, os, sys, glob
import numpy as np, torch, cv2
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = "/home/vnw/tilebac_reanalysis"
PRED = f"{ROOT}/out/predictions"
WORK = f"{ROOT}/work"
DEV = torch.device("cuda")
COCODIR = {"640":  f"{ROOT}/data/uldm_bm/annotated/640p-COCO/BCET Montage 2048-Tile.v9i.coco-segmentation",
           "1024": f"{ROOT}/data/uldm_bm/annotated/1024p-COCO/BCET Montage 2048-Tile.v8i.coco-segmentation"}
YOLODIR = {"640":  f"{ROOT}/data/uldm_bm/annotated/640p-YOLO/BCET Montage 2048-Tile.v9i.yolov11",
           "1024": f"{ROOT}/data/uldm_bm/annotated/1024p-YOLO/BCET Montage 2048-Tile.v8i.yolov11"}
SPLIT = "valid"

def gt_path(res):
    p = f"{WORK}/gt_{res}_{SPLIT}.json"
    if not os.path.exists(p):
        j = json.load(open(f"{COCODIR[res]}/{SPLIT}/_annotations.coco.json"))
        j["categories"] = [c for c in j["categories"] if c["id"] != 0]
        json.dump(j, open(p, "w"))
    return p

def encode(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii"); return r

def save(preds, arch, res):
    p = f"{PRED}/{arch}_{res}_{SPLIT}.json"
    json.dump(preds, open(p, "w"))
    print(f"  -> {os.path.basename(p)} ({len(preds)} instances)", flush=True)

# Dect2
def run_d2(res):
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = f"{ROOT}/ckpt/{res}-Detectron2-ULDM.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.INPUT.MIN_SIZE_TEST = int(res); cfg.INPUT.MAX_SIZE_TEST = int(res)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    model = build_model(cfg); DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(DEV).eval()
    coco = COCO(gt_path(res)); preds = []
    with torch.no_grad():
        for iid in sorted(coco.getImgIds()):
            info = coco.imgs[iid]
            im = cv2.imread(f"{COCODIR[res]}/{SPLIT}/{info['file_name']}")
            H, W = im.shape[:2]
            out = model([{"image": torch.as_tensor(im.transpose(2,0,1)).float().to(DEV),
                          "height": H, "width": W}])[0]["instances"].to("cpu")
            for i in range(len(out)):
                preds.append(dict(image_id=int(iid), category_id=int(out.pred_classes[i].item())+1,
                                  segmentation=encode(out.pred_masks[i].numpy()),
                                  score=float(out.scores[i].item())))
    save(preds, "Detectron2", res)

# unet
def run_unet(res):
    import importlib.util
    from PIL import Image
    import torchvision.transforms as T
    mp = f"{ROOT}/repos/Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets/src/model.py"
    spec = importlib.util.spec_from_file_location("unet_model", mp)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    model = mod.UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(f"{ROOT}/ckpt/{res}-U-Net-ULDM.pt", map_location="cpu"))
    model.to(DEV).eval()
    coco = COCO(gt_path(res)); preds = []
    with torch.no_grad():
        for iid in sorted(coco.getImgIds()):
            info = coco.imgs[iid]
            im = Image.open(f"{COCODIR[res]}/{SPLIT}/{info['file_name']}").convert("L")
            x = T.ToTensor()(im).unsqueeze(0).to(DEV)
            prob = torch.sigmoid(model(x))[0].cpu().numpy()
            for ch in (0, 1):
                n, lab = cv2.connectedComponents((prob[ch] > 0.5).astype(np.uint8))
                for k in range(1, n):
                    comp = lab == k
                    if comp.sum() < 10: continue
                    preds.append(dict(image_id=int(iid), category_id=ch+1,
                                      segmentation=encode(comp),
                                      score=float(prob[ch][comp].mean())))
    save(preds, "U-Net", res)

# yolo
def run_yolo(res):
    from ultralytics import YOLO
    coco = COCO(gt_path(res))
    stem2id = {os.path.splitext(v["file_name"])[0]: k for k, v in coco.imgs.items()}
    for arch in ["YOLOv11", "YOLO26"]:
        m = YOLO(f"{ROOT}/ckpt/{res}-{arch}-ULDM.pt")
        m.val(data=f"{WORK}/yolo_{res}.yaml", split="val", imgsz=int(res), verbose=False,
              plots=False, save_json=True, project=f"{WORK}/valid_json", name=f"{arch}_{res}",
              exist_ok=True)
        src = glob.glob(f"/home/vnw/acorn/runs/segment/{WORK.split('/')[-1]}/valid_json/{arch}_{res}/predictions.json")
        src += glob.glob(f"{WORK}/valid_json/{arch}_{res}/predictions.json")
        src += glob.glob(f"/home/vnw/acorn/runs/segment/**/valid_json/{arch}_{res}/predictions.json", recursive=True)
        src = [s for s in src if os.path.exists(s)]
        assert src, f"no predictions.json for {arch} {res}"
        j = json.load(open(src[0]))
        out, miss = [], 0
        for x in j:
            iid = stem2id.get(x["image_id"])
            if iid is None: miss += 1; continue
            out.append(dict(image_id=int(iid), category_id=int(x["category_id"]),
                            segmentation=x["segmentation"], score=float(x["score"])))
        assert miss == 0, f"{arch} {res}: {miss} unmapped"
        save(out, arch, res)

if __name__ == "__main__":
    which = sys.argv[1]
    for res in ["640", "1024"]:
        print(f"[{which} {res}]", flush=True)
        {"d2": run_d2, "unet": run_unet, "yolo": run_yolo}[which](res)
