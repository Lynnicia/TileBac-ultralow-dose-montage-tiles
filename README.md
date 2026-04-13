Background

Overview
This dataset contains ultralow-dose cryoEM montage tile images of the bacteria Pantoea sp. YR343. Segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 have been fine-tuned to predict bacterial inner membranes and outer membranes. This bacterial membrane dataset is a benchmark dataset to challenge current AI workflows in rapid, seamless montage stitching and stitched segmentation in extremely noisy ultralow-dose cryoEM images. Bacterial flagella low-dose cryoEM images have also been added as a dataset challenge to segmenting high-boundary thin objects in noisy low-dose cryoEM images.
Dataset and Model Repository

Datasets and raw cryoEM images can be found on https://github.com/Lynnicia/TileBac-ultralow-dose-montage-tiles as well as on Hugging Face at: https://huggingface.co/datasets/LynnMass/tilebac-ULDM-benchmark-dataset, https://huggingface.co/datasets/LynnMass/tilebac-ULDM-tiles, https://huggingface.co/datasets/LynnMass/tilebac-stitched-montage and https://huggingface.co/datasets/LynnMass/tilebac-flag-dataset. 

Models can be found on Hugging Face at: https://huggingface.co/LynnMass/640-SAM3-ULDM, https://huggingface.co/LynnMass/1024-SAM3-ULDM, https://huggingface.co/LynnMass/1024-YOLO26-ULDM, https://huggingface.co/LynnMass/640-YOLO26-ULDM, https://huggingface.co/LynnMass/1024-YOLOv11-ULDM, https://huggingface.co/LynnMass/640-YOLOv11-ULDM, https://huggingface.co/LynnMass/1024-U-Net-ULDM, https://huggingface.co/LynnMass/640-U-Net-ULDM, https://huggingface.co/LynnMass/1024-Detectron2-ULDM and https://huggingface.co/LynnMass/640-Detectron2-ULDM. 

SAM3 model placeholders have been added to this dataset. Please visit the links above for the full SAM3 distribution.


