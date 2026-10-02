Background

Overview
This dataset contains ultralow-dose cryoEM montage tile images of the bacteria Pantoea sp. YR343. Segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 have been fine-tuned to predict bacterial inner membranes and outer membranes. This bacterial membrane dataset is a benchmark dataset to challenge current AI workflows in rapid segmentation in extremely noisy ultralow-dose cryoEM images. Bacterial flagella low-dose cryoEM images have also been added as a dataset challenge to segmenting high-boundary thin objects in noisy low-dose cryoEM images.

Dataset and Model Repository

All datasets are openly available on Constellation (DOI: 10.13139/ORNLNCCS/3025229). 

Datasets are located on Hugging Face at: https://huggingface.co/datasets/LynnMass/tilebac-ULDM-benchmark-dataset. 

Models (3 seed) can be found on Hugging Face at: https://huggingface.co/buckets/LynnMass/tilebac_benchmark_models_3seeds 

Reference

Please cite this Biorxiv paper in association with this dataset, the Bibtex for the associated paper is below:
```
@article {Massenburg2026.06.08.731030,
	author = {Massenburg, Lynnicia N. and Madugula, Sita S. and Brown, Spenser R. and Bible, Amber N. and Harris, Chanda R. and Retterer, Scott T. and Morrell-Falvey, Jennifer L. and Vasudevan, Rama K. and Williams, Alexis N.},
	title = {TileBac: A Benchmark CryoEM Dataset of Bacteria in Ultralow-Dose Montage Tiles},
	elocation-id = {2026.06.08.731030},
	year = {2026},
	doi = {10.64898/2026.06.08.731030},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Current segmentation models are capable of routine identification of biological features in noisy cryogenic electron microscopy (cryoEM) images. However, there are still challenges with complete segmentation of high boundary, thin objects such as bacterial cell envelopes and flagella. Moreover, ultralow-dose cryoEM images pose as an additional challenge to boundary distinctions between the object and background. Here, we present TileBac, a benchmark dataset of ultralow-dose montage tiles of Pantoea sp. YR343 to segment bacterial inner and outer membranes for evaluation of model effectiveness. We show that foundation models outperform convolutional neural networks at continuous bacterial cell envelope segmentation despite having lower performance metrics. We release the TileBac benchmark dataset on Hugging Face for further insights into model architecture development.Competing Interest StatementThe authors have declared no competing interest.U.S. Department of Energy, Office of Science, https://ror.org/00mmn6b08, FWP ERKCZ64UT-Battelle, LLC, https://ror.org/04nza6677, DE-AC05- 00OR22725Oak Ridge National Laboratory, https://ror.org/01qz5mb56},
	URL = {https://www.biorxiv.org/content/early/2026/06/09/2026.06.08.731030},
	eprint = {https://www.biorxiv.org/content/early/2026/06/09/2026.06.08.731030.full.pdf},
	journal = {bioRxiv}
}
```

## Unified Evaluation Pipeline

The `unified_evaluation/` directory consolidates the original model-specific evaluation scripts into a reproducible evaluation workflow for YOLOv11, YOLO26, U-Net, Detectron2, and SAM3 at 640 and 1024 resolution.

### Download dataset and model checkpoints

The released TileBac dataset and trained model checkpoints are not stored directly in this Git repository. Download them automatically with:

```bash
python download_assets.py
```

The script downloads the released assets into:

```text
data/uldm_bm/       # TileBac benchmark dataset
checkpoints/        # trained model checkpoints
```

These directories are excluded from Git.

### Environment setup

The unified evaluation pipeline was tested with Python 3.12 on an NVIDIA A100 GPU with CUDA 12.8.

Create and activate the environment:

```bash
python3.12 -m venv tilebac_eval
source tilebac_eval/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch with CUDA 12.8 support:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Install the standard dependencies:

```bash
pip install -r requirements.txt
```

Install Detectron2:

```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

Install SAM3:

```bash
pip install 'git+https://github.com/facebookresearch/sam3.git'
pip install "setuptools<81"
pip install einops
```

The U-Net evaluation uses the model definition from the original `Semantic-Segmentation-of-bacterial-cell-envelope-using-U-Nets` repository. Clone that repository into `repos/` before running U-Net inference.

### Run the evaluation

Prepare the test and validation ground truth:

```bash
python unified_evaluation/01_prepare_ground_truth.py --split test --resolution all
python unified_evaluation/01_prepare_ground_truth.py --split valid --resolution all
```

Run model inference using `02_predict.py`, followed by the common COCO evaluation and diagnostic scripts in numerical order.

```bash
python unified_evaluation/02_predict.py --help
python unified_evaluation/03_coc_evaluation.py --resolution all
python unified_evaluation/04_pixel_f1.py
```

Additional analyses are provided in `unified_evaluation/diagnostic/`.

`10_verify_claims.py` is retained as an optional historical/manuscript self-audit. Some checks depend on auxiliary intermediate files from the original analysis that are not included with the released benchmark assets, so this script is not required for the core evaluation workflow.
