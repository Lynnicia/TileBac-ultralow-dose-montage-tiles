"""
Generate model predictions for unified COCO evaluation.
Models:YOLOv11,YOLO26,Detectron2,U-Net,SAM3

Predictions are written in common COCO format given below:
    {
        "image_id": int,
        "category_id": int,
        "segmentation": COCO_RLE,
        "score": float
    }
The existing TileBac validation/test partitions are used unchanged.
"""

import argparse
import glob
import importlib.util
import json
import os

import cv2
import numpy as np
import torch

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    WORK_DIR,
    PREDICTION_DIR,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET PATHS
# ============================================================

COCO_DIR = {
    "640": (
        DATA_DIR
        / "uldm_bm"
        / "annotated"
        / "640p-COCO"
        / "BCET Montage 2048-Tile.v9i.coco-segmentation"
    ),
    "1024": (
        DATA_DIR
        / "uldm_bm"
        / "annotated"
        / "1024p-COCO"
        / "BCET Montage 2048-Tile.v8i.coco-segmentation"
    ),
}

YOLO_DIR = {
    "640": (
        DATA_DIR
        / "uldm_bm"
        / "annotated"
        / "640p-YOLO"
        / "BCET Montage 2048-Tile.v9i.yolov11"
    ),
    "1024": (
        DATA_DIR
        / "uldm_bm"
        / "annotated"
        / "1024p-YOLO"
        / "BCET Montage 2048-Tile.v8i.yolov11"
    ),
}


# ============================================================
# COMMON HELPERS
# ============================================================

def gt_path(res, split):
    """Return canonical COCO ground-truth file prepared by 01."""

    if split == "test":
        path = WORK_DIR / f"gt_{res}.json"
    else:
        path = WORK_DIR / f"gt_{res}_{split}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n{path}\n\n"
            f"Run 01_prepare_ground_truth.py --split {split} first."
        )
    return path


def image_dir(resolution, split):
    """Existing COCO image partition."""
    return COCO_DIR[resolution] / split


def encode(mask):
    """Encode a binary mask as COCO RLE."""
    rle = mask_utils.encode(
        np.asfortranarray(mask.astype(np.uint8))
    )
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def save_predictions(predictions, architecture, resolution, split):
    """Write predictions in common COCO format."""
    path = (
        PREDICTION_DIR
        / f"{architecture}_{resolution}_{split}.json"
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(
        f"  -> {path.name} "
        f"({len(predictions)} instances)",
        flush=True,
    )


def load_images(resolution, split):
    coco = COCO(str(gt_path(resolution, split)))

    images = [
        (image_id, coco.imgs[image_id])
        for image_id in sorted(coco.getImgIds())
    ]

    return coco, images


# ============================================================
# YOLO
# ============================================================

def run_yolo_test(architecture, resolution):
    """
    Test-set YOLO prediction.

    Preserves the original 06_predict.py inference procedure:
    model.predict() image-by-image, followed by COCO RLE encoding.
    """
    from ultralytics import YOLO

    print(
        f"[{architecture} {resolution} test]",
        flush=True,
    )

    checkpoint = (
        CHECKPOINT_DIR
        / f"{resolution}-{architecture}-ULDM.pt"
    )

    model = YOLO(str(checkpoint))

    coco, images = load_images(
        resolution,
        "test",
    )

    predictions = []

    for image_id, info in images:

        image_path = (
            image_dir(resolution, "test")
            / info["file_name"]
        )

        result = model.predict(
            str(image_path),
            imgsz=int(resolution),
            conf=0.001,
            iou=0.7,
            max_det=300,
            retina_masks=True,
            verbose=False,
            device=0,
        )[0]

        if result.masks is None:
            continue

        height = info["height"]
        width = info["width"]

        for mask, category, score in zip(
            result.masks.data.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
            result.boxes.conf.cpu().numpy(),
        ):

            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category) + 1,
                    "segmentation": encode(mask > 0.5),
                    "score": float(score),
                }
            )

    save_predictions(
        predictions,
        architecture,
        resolution,
        "test",
    )


def run_yolo_valid(resolution):
    """
    Validation-set YOLO prediction.

    Preserves the native Ultralytics COCO export used in
    08_predict_valid.py.
    """
    from ultralytics import YOLO

    coco = COCO(
        str(gt_path(resolution, "valid"))
    )

    stem_to_id = {
        os.path.splitext(info["file_name"])[0]: image_id
        for image_id, info in coco.imgs.items()
    }

    for architecture in ["YOLOv11", "YOLO26"]:

        print(
            f"[{architecture} {resolution} valid]",
            flush=True,
        )

        checkpoint = (
            CHECKPOINT_DIR
            / f"{resolution}-{architecture}-ULDM.pt"
        )

        model = YOLO(str(checkpoint))

        output_dir = (
            WORK_DIR
            / "valid_json"
        )

        run_name = f"{architecture}_{resolution}"

        model.val(
            data=str(WORK_DIR / f"yolo_{resolution}.yaml"),
            split="val",
            imgsz=int(resolution),
            verbose=False,
            plots=False,
            save_json=True,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
        )

        prediction_json = (
            output_dir
            / run_name
            / "predictions.json"
        )

        if not prediction_json.exists():
            raise FileNotFoundError(
                "Ultralytics predictions.json not found:\n"
                f"{prediction_json}"
            )

        with prediction_json.open(
            "r",
            encoding="utf-8",
        ) as f:
            native_predictions = json.load(f)

        predictions = []
        missing = 0

        for prediction in native_predictions:

            image_id = stem_to_id.get(
                str(prediction["image_id"])
            )

            if image_id is None:
                missing += 1
                continue

            predictions.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(
                        prediction["category_id"]
                    ),
                    "segmentation": prediction[
                        "segmentation"
                    ],
                    "score": float(
                        prediction["score"]
                    ),
                }
            )

        assert missing == 0, (
            f"{architecture} {resolution}: "
            f"{missing} unmapped image IDs"
        )

        save_predictions(
            predictions,
            architecture,
            resolution,
            "valid",
        )


def run_yolo(resolution, split):
    """Run the original YOLO procedure for the requested split."""

    if split == "test":

        for architecture in [
            "YOLOv11",
            "YOLO26",
        ]:
            run_yolo_test(
                architecture,
                resolution,
            )

    else:
        run_yolo_valid(
            resolution,
        )

# ============================================================
# DETECTRON2
# ============================================================

def run_detectron2(resolution, split):

    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    print(
        f"[Detectron2 {resolution} {split}]",
        flush=True,
    )

    checkpoint = (
        CHECKPOINT_DIR
        / f"{resolution}-Detectron2-ULDM.pth"
    )

    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/"
            "mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg.INPUT.MIN_SIZE_TEST = int(resolution)
    cfg.INPUT.MAX_SIZE_TEST = int(resolution)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001

    model = build_model(cfg)

    DetectionCheckpointer(model).load(
        cfg.MODEL.WEIGHTS
    )

    model.to(DEVICE).eval()

    coco, images = load_images(
        resolution,
        split,
    )

    predictions = []

    with torch.no_grad():

        for image_id, info in images:

            image_path = (
                image_dir(resolution, split)
                / info["file_name"]
            )

            image = cv2.imread(str(image_path))

            H, W = image.shape[:2]

            output = model(
                [
                    {
                        "image": torch.as_tensor(
                            image.transpose(2, 0, 1)
                        )
                        .float()
                        .to(DEVICE),
                        "height": H,
                        "width": W,
                    }
                ]
            )[0]["instances"].to("cpu")

            for i in range(len(output)):

                predictions.append(
                    {
                        "image_id": int(image_id),

                        "category_id": int(
                            output.pred_classes[i].item()
                        ) + 1,

                        "segmentation": encode(
                            output.pred_masks[i].numpy()
                        ),

                        "score": float(
                            output.scores[i].item()
                        ),
                    }
                )

    save_predictions(
        predictions,
        "Detectron2",
        resolution,
        split,
    )


# ============================================================
# U-NET
# ============================================================

def run_unet(resolution, split):
    """
    U-Net semantic segmentation -> instances using connected
    components.

    This preserves the derivation used in the original scripts:
        sigmoid
        threshold > 0.5
        connected components
        remove components < 10 pixels
        component score = mean probability
    """

    from PIL import Image
    import torchvision.transforms as T

    print(
        f"[U-Net {resolution} {split}]",
        flush=True,
    )

    # Preserve the original workflow:
    # load the U-Net architecture from the existing U-Net repo.
    # Expected location:
    #
    # repos: Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets/src/model.py
    
    unet_model_path = (
        DATA_DIR.parent
        / "repos"
        / "Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets"
        / "src"
        / "model.py"
    )

    if not unet_model_path.exists():
        raise FileNotFoundError(
            "U-Net model.py not found:\n"
            f"{unet_model_path}\n\n"
            "The original evaluation code loads the U-Net "
            "architecture from this repository."
        )

    spec = importlib.util.spec_from_file_location(
        "unet_model",
        unet_model_path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.UNet(
        in_channels=1,
        out_channels=2,
    )

    checkpoint = (
        CHECKPOINT_DIR
        / f"{resolution}-U-Net-ULDM.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location="cpu",
        )
    )

    model.to(DEVICE).eval()

    coco, images = load_images(
        resolution,
        split,
    )

    predictions = []

    with torch.no_grad():
        for image_id, info in images:
            image_path = (
                image_dir(resolution, split)
                / info["file_name"]
            )
            image = Image.open(
                image_path
            ).convert("L")
            x = (
                T.ToTensor()(image)
                .unsqueeze(0)
                .to(DEVICE)
            )
            probability = (
                torch.sigmoid(model(x))[0]
                .cpu()
                .numpy()
            )
            # channel 0 = IM
            # channel 1 = OM
            for channel in (0, 1):
                binary_mask = (
                    probability[channel] > 0.5
                ).astype(np.uint8)

                n_components, labels = (
                    cv2.connectedComponents(binary_mask)
                )
                for component_id in range(
                    1,
                    n_components,
                ):
                    component = (
                        labels == component_id
                    )
                    # Preserve original speck removal.
                    if component.sum() < 10:
                        continue

                    predictions.append(
                        {
                            "image_id": int(image_id),

                            "category_id": channel + 1,

                            "segmentation": encode(
                                component
                            ),

                            "score": float(
                                probability[channel][
                                    component
                                ].mean()
                            ),
                        }
                    )

    save_predictions(
        predictions,
        "U-Net",
        resolution,
        split,
    )


# ============================================================
# SAM3
# ============================================================

def make_sam3_datapoint(pil_image, prompts):

    from sam3.train.data.sam3_image_dataset import (
        InferenceMetadata,
        FindQueryLoaded,
        Image as SAMImage,
        Datapoint,
    )

    datapoint = Datapoint(
        find_queries=[],
        images=[],
    )

    width, height = pil_image.size

    datapoint.images = [
        SAMImage(
            data=pil_image,
            objects=[],
            size=[height, width],
        )
    ]

    prompt_ids = {}

    for i, prompt in enumerate(prompts):

        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=i,
                    original_image_id=i,
                    original_category_id=1,
                    original_size=[width, height],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )

        prompt_ids[prompt] = i

    return datapoint, prompt_ids


def run_sam3(resolution, split):

    from PIL import Image

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.utils.misc import copy_data_to_device

    from sam3.train.data.collator import (
        collate_fn_api as collate
    )

    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        RandomResizeAPI,
        ToTensorAPI,
        NormalizeAPI,
    )

    from sam3.eval.postprocessors import (
        PostProcessImage
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The original SAM3 evaluation workflow requires CUDA."
        )

    sam_device = torch.device("cuda")

    print(
        f"[SAM3 {resolution} {split}] building model",
        flush=True,
    )

    bpe_path = (
        WORK_DIR
        / "bpe_simple_vocab_16e6.txt.gz"
    )

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        enable_segmentation=True,
        eval_mode=False,
        load_from_HF=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR
        / f"{resolution}-SAM3-ULDM.pt",
        map_location="cpu",
        weights_only=False,
    )

    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict)
        and "model" in checkpoint
        else checkpoint
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"  loaded state_dict: "
        f"{len(missing)} missing, "
        f"{len(unexpected)} unexpected",
        flush=True,
    )

    model = model.to(
        sam_device
    ).eval()

    del checkpoint, state_dict

    coco = COCO(
        str(gt_path(resolution, split))
    )

    category_name_to_id = {
        category["name"]: category["id"]
        for category in coco.dataset["categories"]
    }

    predictions = []

    image_ids = sorted(
        coco.getImgIds()
    )

    for n, image_id in enumerate(image_ids):

        info = coco.imgs[image_id]

        image_path = (
            image_dir(resolution, split)
            / info["file_name"]
        )

        pil_image = Image.open(
            image_path
        ).convert("RGB")

        width, height = pil_image.size

        with (
            torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ),
            torch.no_grad(),
        ):

            datapoint, prompt_ids = (
                make_sam3_datapoint(
                    pil_image,
                    ["IM", "OM"],
                )
            )

            datapoint = transform(
                datapoint
            )

            batch = collate(
                [datapoint],
                dict_key="data",
            )["data"]

            batch = copy_data_to_device(
                batch,
                sam_device,
                non_blocking=True,
            )

            output = model(batch)

            postprocessor = PostProcessImage(
                max_dets_per_img=-1,
                iou_type="segm",
                use_original_sizes_box=True,
                use_original_sizes_mask=True,
                convert_mask_to_rle=False,
                detection_threshold=0.001,
                to_cpu=True,
            )

            results = postprocessor.process_results(
                output,
                batch.find_metadatas,
            )

        for prompt in ["IM", "OM"]:

            category_id = (
                category_name_to_id[prompt]
            )

            result = results[
                prompt_ids[prompt]
            ]

            for mask, score in zip(
                result["masks"],
                result["scores"],
            ):

                mask = (
                    mask.float()
                    .numpy()
                    .squeeze()
                )

                if mask.shape != (
                    height,
                    width,
                ):
                    mask = cv2.resize(
                        mask,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                binary_mask = (
                    mask > 0.5
                ).astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "segmentation": encode(
                            binary_mask
                        ),
                        "score": float(score),
                    }
                )

        del batch, output
        torch.cuda.empty_cache()

        if (n + 1) % 10 == 0:
            print(
                f"  {n + 1}/{len(image_ids)} "
                f"preds={len(predictions)}",
                flush=True,
            )

    save_predictions(
        predictions,
        "SAM3",
        resolution,
        split,
    )

    del model
    torch.cuda.empty_cache()


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate TileBac predictions for "
            "uniform COCO evaluation."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "all",
            "yolo",
            "detectron2",
            "unet",
            "sam3",
        ],
        default="all",
    )

    parser.add_argument(
        "--split",
        choices=[
            "valid",
            "test",
        ],
        required=True,
    )

    parser.add_argument(
        "--resolution",
        choices=[
            "640",
            "1024",
            "all",
        ],
        default="all",
    )

    args = parser.parse_args()

    resolutions = (
        ["640", "1024"]
        if args.resolution == "all"
        else [args.resolution]
    )

    for resolution in resolutions:

        if args.model in (
            "all",
            "yolo",
        ):
            run_yolo(
                resolution,
                args.split,
            )

        if args.model in (
            "all",
            "detectron2",
        ):
            run_detectron2(
                resolution,
                args.split,
            )

        if args.model in (
            "all",
            "unet",
        ):
            run_unet(
                resolution,
                args.split,
            )

        if args.model in (
            "all",
            "sam3",
        ):
            run_sam3(
                resolution,
                args.split,
            )


if __name__ == "__main__":
    main()