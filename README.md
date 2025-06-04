# GelSight Smoothness Classification

This project contains a minimal script for classifying the surface smoothness of materials using images captured from a GelSight sensor.

## Dataset structure

The training code expects a directory containing subfolders named `material_<ID>` where `<ID>` is an integer label starting from 1. Each material folder contains one or more `cycle_*` folders with a `frame.png` image:

```
dataset_root/
  material_1/
    cycle_001/frame.png
    cycle_002/frame.png
  material_2/
    cycle_003/frame.png
    ...
```

Provide the dataset root with `--root` when running the script.

## Training stages

The `torch_smooth.py` script runs two phases:

1. **Self-supervised pretraining** (SimCLR) on image tiles using an EfficientNet-B6 backbone. This stage learns representations without labels.
2. **Supervised fine‑tuning** on the labeled tiles with cross‑entropy loss and K‑fold cross validation.

Running `python torch_smooth.py --stage all --root <dataset>` executes both stages sequentially.

## Required packages

The code relies on Python 3 with the following packages:

- torch
- torchvision
- lightly
- numpy
- pillow
- scikit-learn
- torchmetrics
- tqdm

Install these through `pip` before training.
