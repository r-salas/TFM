#
#
#   Utils
#
#

import os
import gzip
import torch
import shutil
import zipfile

from tqdm import tqdm
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
