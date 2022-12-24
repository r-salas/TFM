#!/bin/bash


# Download datasets
python download_chest_xrays_indiana_university.py
python download_chexpert.py
python download_mini_imagenet.py
python download_nih_chest_xrays.py
python download_padchest.py
python download_unifesp_xray_body_part.py

# Download RadImageNet weights
python download_radimagenet_weights.py
python radimagenet_to_onnx.py
