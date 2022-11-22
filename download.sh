#!/bin/bash


# Download datasets
python download_chest_xrays_indiana_university.py
python download_chexpert.py
python download_nih_chest_xrays.py
python download_padchest.py

# Download RadImageNet weights
python download_radimagenet_weights.py
python radimagenet_to_onnx.py
