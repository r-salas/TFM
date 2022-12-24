#
#
#   UNIFESP X-ray body part
#
#

import os
import io
import h5py
import kaggle
import zipfile
import warnings

import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from utils import read_dicom, remove_suffix
from config import DOWNLOADS_ROOT_DIR, DATA_ROOT_DIR


download_path = os.path.join(DOWNLOADS_ROOT_DIR, "unifesp-x-ray-body-part-classifier")

kaggle.api.competition_download_files("unifesp-x-ray-body-part-classifier", download_path, quiet=False)

with (zipfile.ZipFile(os.path.join(download_path, "unifesp-x-ray-body-part-classifier.zip")) as archive,
      h5py.File(os.path.join(DATA_ROOT_DIR, "unifesp-x-ray-body-part-classifier",
                             "unifesp-x-ray-body-part-classifier.h5"), "w") as hdf5_file):
    meta_data = archive.read("train.csv")
    meta_bytes = io.BytesIO(meta_data)
    meta = pd.read_csv(meta_bytes).set_index("SOPInstanceUID")

    train_dcm_paths = []
    for archive_file in archive.infolist():
        if archive_file.filename.startswith("train/") and archive_file.filename.endswith(".dcm"):
            train_dcm_paths.append(archive_file.filename)

    data = hdf5_file.create_dataset("data", shape=(len(train_dcm_paths), 256, 256), dtype="uint8")
    labels = hdf5_file.create_dataset("labels", shape=(len(train_dcm_paths)), dtype=h5py.special_dtype(vlen=str))

    for index, dcm_path in enumerate(tqdm(train_dcm_paths)):
        img_data = archive.read(dcm_path)
        img_bytes = io.BytesIO(img_data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            np_img = read_dicom(img_bytes)

        pil_img = Image.fromarray(np_img).convert("L")
        pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)

        data[index] = np.asarray(pil_img).astype("uint8")

        _, fname = os.path.split(dcm_path)
        uid = remove_suffix(fname, "-c.dcm")

        labels[index] = meta.loc[uid, "Target"].strip().replace(" ", ";")
