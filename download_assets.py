#!/usr/bin/env python3

"""
Download the released TileBac benchmark dataset and trained model checkpoints
used by the unified evaluation pipeline.

Assets are downloaded from the LynnMass Hugging Face repositories into:

    data/uldm_bm/
    checkpoints/

These directories are intentionally excluded from Git.
"""

from pathlib import Path

from huggingface_hub import snapshot_download, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data" / "uldm_bm"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


# ---------------------------------------------------------------------
# Released TileBac dataset
# ---------------------------------------------------------------------

DATASET_REPO = "LynnMass/tilebac-ULDM-benchmark-dataset"


# ---------------------------------------------------------------------
# Released model checkpoints
# repo_id : checkpoint filename
# ---------------------------------------------------------------------

CHECKPOINTS = {
    "LynnMass/640-YOLOv11-ULDM": "640-YOLOv11-ULDM.pt",
    "LynnMass/1024-YOLOv11-ULDM": "1024-YOLOv11-ULDM.pt",

    "LynnMass/640-YOLO26-ULDM": "640-YOLO26-ULDM.pt",
    "LynnMass/1024-YOLO26-ULDM": "1024-YOLO26-ULDM.pt",

    "LynnMass/640-Detectron2-ULDM": "640-Detectron2-ULDM.pth",
    "LynnMass/1024-Detectron2-ULDM": "1024-Detectron2-ULDM.pth",

    "LynnMass/640-U-Net-ULDM": "640-U-Net-ULDM.pt",
    "LynnMass/1024-U-Net-ULDM": "1024-U-Net-ULDM.pt",

    "LynnMass/640-SAM3-ULDM": "640-SAM3-ULDM.pt",
    "LynnMass/1024-SAM3-ULDM": "1024-SAM3-ULDM.pt",
}


def download_dataset():
    """Download the released TileBac benchmark dataset."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("Downloading TileBac benchmark dataset")
    print(f"Repository : {DATASET_REPO}")
    print(f"Destination: {DATA_DIR}")
    print("============================================================\n")

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )


def download_checkpoints():
    """Download all released model checkpoints."""

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n============================================================")
    print("Downloading trained model checkpoints")
    print(f"Destination: {CHECKPOINT_DIR}")
    print("============================================================\n")

    for repo_id, filename in CHECKPOINTS.items():

        destination = CHECKPOINT_DIR / filename

        if destination.exists():
            print(f"[EXISTS] {filename}")
            continue

        print(f"[DOWNLOAD] {repo_id} -> {filename}")

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=CHECKPOINT_DIR,
        )


def main():

    print("\nTileBac evaluation asset downloader")
    print("-----------------------------------")
    print(f"Repository root: {REPO_ROOT}")

    download_dataset()
    download_checkpoints()

    print("\n============================================================")
    print("TileBac assets ready")
    print("============================================================")
    print(f"Dataset     : {DATA_DIR}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print("\nYou can now run the unified evaluation pipeline.")


if __name__ == "__main__":
    main()
