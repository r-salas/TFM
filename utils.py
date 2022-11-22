#
#
#   Utils
#
#

import os
import gzip

import numpy as np
import torch
import shutil
import zipfile
import pandas as pd

from tqdm import tqdm
from typing import Optional
from PIL import Image, ImageMath


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def extract_gz(path, target_dir):
    fname, _ = os.path.splitext(os.path.basename(path))
    new_path = os.path.join(target_dir, fname)

    with gzip.open(path, "rb") as f_in:
        with open(new_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return new_path


def extract_zip(path, target_dir, to_extract=None):
    fname = os.path.basename(path)
    with zipfile.ZipFile(path) as zf:
        for file in tqdm(zf.namelist(), desc=f"Extracting {fname}..."):
            if os.path.isfile(os.path.join(target_dir, file)):
                continue
            if (to_extract is None or file in to_extract or any([file.startswith(pattern) for pattern in to_extract
                                                                 if "/" in pattern])):
                zf.extract(file, target_dir)


def convert_I_to_L(im: Image):
    return ImageMath.eval('im >> 8', im=im.convert('I')).convert('L')


def undersample(df, label_col, random_state: Optional[int] = None):
    min_samples = df.groupby(label_col).size().min()

    undersample_data = []
    for label, label_df in df.groupby(label_col):
        label_df = label_df.sample(min_samples, random_state=random_state)
        undersample_data.append(label_df)

    return pd.concat(undersample_data)


def pil_grayscale_to_rgb(pil_img):
    return pil_img.convert("RGB")


def radimagenet_transforms(pil_img):
    pil_img = pil_img.resize((224, 224), Image.ANTIALIAS)
    img = np.asarray(pil_img)
    img = (img - 127.5) * 2 / 255

    if img.ndim == 2:
        img = img[:, :, None]

    return torch.from_numpy(img).contiguous().to(torch.float32)
