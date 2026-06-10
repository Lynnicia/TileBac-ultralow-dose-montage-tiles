Background

Overview
This dataset contains ultralow-dose cryoEM montage tile images of the bacteria Pantoea sp. YR343. Segmentation models from YOLOv11, YOLO26, U-Net, Detectron2 and SAM3 have been fine-tuned to predict bacterial inner membranes and outer membranes. This bacterial membrane dataset is a benchmark dataset to challenge current AI workflows in rapid, seamless montage stitching and stitched segmentation in extremely noisy ultralow-dose cryoEM images. Bacterial flagella low-dose cryoEM images have also been added as a dataset challenge to segmenting high-boundary thin objects in noisy low-dose cryoEM images.

Dataset and Model Repository

All datasets are openly available on Constellation (DOI: 10.13139/ORNLNCCS/3025229). 

Datasets and raw cryoEM images can be found on https://github.com/Lynnicia/TileBac-ultralow-dose-montage-tiles as well as on Hugging Face at: https://huggingface.co/datasets/LynnMass/tilebac-ULDM-benchmark-dataset, https://huggingface.co/datasets/LynnMass/tilebac-ULDM-tiles, https://huggingface.co/datasets/LynnMass/tilebac-stitched-montage and https://huggingface.co/datasets/LynnMass/tilebac-flag-dataset. 

Models can be found on Hugging Face at: https://huggingface.co/LynnMass/640-SAM3-ULDM, https://huggingface.co/LynnMass/1024-SAM3-ULDM, https://huggingface.co/LynnMass/1024-YOLO26-ULDM, https://huggingface.co/LynnMass/640-YOLO26-ULDM, https://huggingface.co/LynnMass/1024-YOLOv11-ULDM, https://huggingface.co/LynnMass/640-YOLOv11-ULDM, https://huggingface.co/LynnMass/1024-U-Net-ULDM, https://huggingface.co/LynnMass/640-U-Net-ULDM, https://huggingface.co/LynnMass/1024-Detectron2-ULDM and https://huggingface.co/LynnMass/640-Detectron2-ULDM. 

SAM3 model placeholders have been added to this dataset. Please visit the links above for the full SAM3 distribution.

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
