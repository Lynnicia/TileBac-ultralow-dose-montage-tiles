"""
Prepare COCO ground-truth files for unified TileBac evaluation.

This uses the existing released validation/test partitions.No new split or annotation is created.
Evaluation classes:
    IM = 1
    OM = 2
"""

import argparse
import json

from config import DATA_DIR, WORK_DIR


COCO_DATASETS = {
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


def prepare_ground_truth(resolution, split):
    """Prepare the existing COCO GT for unified evaluation."""

    dataset_dir = COCO_DATASETS[resolution] / split
    source_json = dataset_dir / "_annotations.coco.json"

    if not source_json.exists():
        raise FileNotFoundError(
            f"COCO annotation file not found:\n{source_json}"
        )

    with source_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Preserve the original evaluation convention: remove unused category ID 0; IM=1 and OM=2 remain unchanged.
    data["categories"] = [
        category
        for category in data["categories"]
        if category["id"] != 0
    ]
    
    if split == "test":
        output_json = WORK_DIR / f"gt_{resolution}.json"
    else:
        output_json = WORK_DIR / f"gt_{resolution}_{split}.json"
    #output_json = WORK_DIR / f"gt_{resolution}_{split}.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Prepared: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TileBac COCO ground truth."
    )

    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        required=True,
        help="Existing TileBac partition to prepare.",
    )

    parser.add_argument(
        "--resolution",
        choices=["640", "1024", "all"],
        default="all",
        help="Dataset resolution.",
    )

    args = parser.parse_args()

    resolutions = (
        ["640", "1024"]
        if args.resolution == "all"
        else [args.resolution]
    )

    for resolution in resolutions:
        prepare_ground_truth(resolution, args.split)


if __name__ == "__main__":
    main()