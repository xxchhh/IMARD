# IMARD: Instrument Meter Automatic Reading Dataset and Model

This repository provides the IMARD dataset and the corresponding model implementation for automatic instrument meter reading.

The project is designed for recognizing digital readings from instrument images under complex visual conditions, such as illumination variation, blur, occlusion, and background interference.

## Repository Structure

```text
IMARD/
├── IMARD.rar
├── model.py
├── inference.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The dataset archive is provided as:

```text
IMARD.rar
```

After downloading or cloning this repository, users can extract `IMARD.rar` to obtain the image data.

The dataset is intended for research on automatic instrument meter reading and digit recognition in complex scenes.

## Model

The released code contains the core model implementation only.

The model consists of a lightweight residual backbone and a quality-aware dynamic fusion mechanism. It includes the following main components:

* Lightweight residual feature extraction backbone
* Image quality estimation module
* Contour/edge enhancement branch
* Context compensation branch
* Dynamic feature fusion module
* Multi-head digit prediction module

The model predicts multiple digit positions simultaneously for instrument meter reading.

## Files

### `model.py`

This file contains the model architecture, including the backbone network, feature enhancement modules, dynamic fusion module, and model creation function.

### `inference.py`

This file provides inference functions for loading a trained model checkpoint and predicting meter readings from input images.

### `requirements.txt`

This file lists the basic Python dependencies required to run the released code.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Prepare a trained model checkpoint and run inference with `inference.py`.

Example:

```bash
python inference.py --checkpoint path/to/model.pth --image path/to/image.jpg
```

For folder-level prediction:

```bash
python inference.py --checkpoint path/to/model.pth --input_dir path/to/images
```

## Notes

This repository releases the model-related source code and the IMARD dataset archive.

Training scripts, dataset splitting details, private training configurations, and evaluation scripts are not included in this public release.

## Citation

If this dataset or code is useful for your research, please consider citing this repository or the related paper when available.

## License

This repository is released for academic and research purposes.
