"""Phase 4 support - SAM3 predictions on the VALIDATION partition."""
import json, os
import numpy as np, torch, cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from sam3.model_builder import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (InferenceMetadata, FindQueryLoaded,
                                                Image as SAMImage, Datapoint)
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.transforms.basic_for_api import (ComposeAPI, RandomResizeAPI,
                                                 ToTensorAPI, NormalizeAPI)
from sam3.eval.postprocessors import PostProcessImage

ROOT = "/home/vnw/tilebac_reanalysis"; PRED = f"{ROOT}/out/predictions"
COCODIR = {"640":  f"{ROOT}/data/uldm_bm/annotated/640p-COCO/BCET Montage 2048-Tile.v9i.coco-segmentation",
           "1024": f"{ROOT}/data/uldm_bm/annotated/1024p-COCO/BCET Montage 2048-Tile.v8i.coco-segmentation"}
DEV = torch.device("cuda"); BPE = f"{ROOT}/work/bpe_simple_vocab_16e6.txt.gz"

def make_datapoint(pil, prompts):
    dp = Datapoint(find_queries=[], images=[]); w, h = pil.size
    dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]; ids = {}
    for i, p in enumerate(prompts):
        dp.find_queries.append(FindQueryLoaded(
            query_text=p, image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(coco_image_id=i, original_image_id=i,
                original_category_id=1, original_size=[w, h], object_id=0, frame_index=0)))
        ids[p] = i
    return dp, ids

transform = ComposeAPI(transforms=[
    RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
    ToTensorAPI(), NormalizeAPI(mean=[0.5]*3, std=[0.5]*3)])

for res in ["640", "1024"]:
    model = build_sam3_image_model(bpe_path=BPE, enable_segmentation=True,
                                   eval_mode=False, load_from_HF=False)
    ck = torch.load(f"{ROOT}/ckpt/{res}-SAM3-ULDM.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    model = model.to(DEV).eval(); del ck
    coco = COCO(f"{ROOT}/work/gt_{res}_valid.json")
    name2id = {c["name"]: c["id"] for c in coco.dataset["categories"]}
    preds = []
    for iid in sorted(coco.getImgIds()):
        info = coco.imgs[iid]
        pil = Image.open(f"{COCODIR[res]}/valid/{info['file_name']}").convert("RGB")
        W, H = pil.size
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            dp, ids = make_datapoint(pil, ["IM", "OM"]); dp = transform(dp)
            batch = copy_data_to_device(collate([dp], dict_key="d")["d"], DEV, non_blocking=True)
            out = model(batch)
            pp = PostProcessImage(max_dets_per_img=-1, iou_type="segm",
                use_original_sizes_box=True, use_original_sizes_mask=True,
                convert_mask_to_rle=False, detection_threshold=0.001, to_cpu=True)
            results = pp.process_results(out, batch.find_metadatas)
        for prompt, cid in [("IM", name2id["IM"]), ("OM", name2id["OM"])]:
            r = results[ids[prompt]]
            for mask, score in zip(r["masks"], r["scores"]):
                m = mask.float().numpy().squeeze()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                b = (m > 0.5).astype(np.uint8)
                if b.sum() == 0: continue
                rle = mask_utils.encode(np.asfortranarray(b)); rle["counts"] = rle["counts"].decode()
                preds.append(dict(image_id=int(iid), category_id=int(cid),
                                  segmentation=rle, score=float(score)))
        del batch, out; torch.cuda.empty_cache()
    json.dump(preds, open(f"{PRED}/SAM3_{res}_valid.json", "w"))
    print(f"  -> SAM3_{res}_valid.json ({len(preds)} instances)", flush=True)
    del model; torch.cuda.empty_cache()
